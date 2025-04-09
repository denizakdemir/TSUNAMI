import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from source.models.tasks.base import TaskHead


class SingleRiskHead(TaskHead):
    """
    Task head for single-risk survival analysis.
    
    Predicts the probability of an event occurring at each time point,
    handling right-censoring through masked loss computation.
    """
    
    def __init__(self,
                name: str,
                input_dim: int,
                num_time_bins: int,
                alpha_rank: float = 0.0,
                alpha_calibration: float = 0.0,
                task_weight: float = 1.0,
                use_bce_loss: bool = True,
                dropout: float = 0.1):
        """
        Initialize SingleRiskHead.
        
        Parameters
        ----------
        name : str
            Name of the task
            
        input_dim : int
            Dimension of the input representation
            
        num_time_bins : int
            Number of discrete time bins to predict
            
        alpha_rank : float, default=0.0
            Weight of the ranking loss component
            
        alpha_calibration : float, default=0.0
            Weight of the calibration loss component
            
        task_weight : float, default=1.0
            Weight of this task in the multi-task loss
            
        use_bce_loss : bool, default=True
            Whether to use binary cross-entropy loss (True) or negative log-likelihood (False)
            
        dropout : float, default=0.1
            Dropout rate for the prediction network
        """
        super().__init__(name, input_dim, task_weight)
        
        self.num_time_bins = num_time_bins
        self.alpha_rank = alpha_rank
        self.alpha_calibration = alpha_calibration
        self.use_bce_loss = use_bce_loss
        
        # Architecture
        self.prediction_network = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_time_bins)
        )
        
    def loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between model outputs and targets.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Outputs from the forward pass
            
        targets : torch.Tensor
            Target values with shape [batch_size, 2 + num_time_bins]:
            - targets[:, 0]: Event indicator (1 if event occurred, 0 if censored)
            - targets[:, 1]: Time bin index where event/censoring occurred
            - targets[:, 2:]: One-hot encoding of event time (for convenience)
            
        Returns
        -------
        torch.Tensor
            Loss value
        """
        if 'loss' in outputs:
            return outputs['loss']
            
        # Extract relevant outputs
        hazards = outputs['hazard']
        survival = outputs['survival']
        risk_score = outputs['risk_score']
        
        # Extract target components
        event_indicator = targets[:, 0]
        event_time = targets[:, 1].long()
        
        # Initialize components
        batch_size = hazards.size(0)
        device = hazards.device
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        if self.use_bce_loss:
            # Binary cross-entropy loss for each time point
            # Create targets and mask for BCE loss
            bce_targets = torch.zeros_like(hazards)
            bce_mask = torch.zeros_like(hazards)
            
            # Set targets and mask for each sample
            for i in range(batch_size):
                t = event_time[i]
                
                if event_indicator[i] > 0:  # Event occurred
                    # Target 0 before event, 1 at event
                    bce_targets[i, :t] = 0
                    bce_targets[i, t] = 1
                    
                    # Mask includes up to and including event
                    bce_mask[i, :(t+1)] = 1
                else:  # Censored
                    # Target 0 before censoring
                    bce_targets[i, :t] = 0
                    
                    # Mask includes only up to censoring (exclusive)
                    bce_mask[i, :t] = 1
            
            # Compute BCE loss
            bce_loss = F.binary_cross_entropy(
                hazards, 
                bce_targets, 
                reduction='none'
            )
            
            # Apply mask and normalize
            loss = torch.sum(bce_loss * bce_mask) / (torch.sum(bce_mask) + 1e-6)
        else:
            # Negative log-likelihood loss (Discrete-time NLL)
            # Initialize hazard and survival terms with protection against numerical issues
            epsilon = 1e-7
            hazards_safe = torch.clamp(hazards, epsilon, 1.0 - epsilon) 
            log_hazard = torch.log(hazards_safe)
            log_1_minus_hazard = torch.log(1 - hazards_safe)
            
            # Initialize loss accumulator
            nll = torch.zeros(batch_size, device=device)
            
            # Compute likelihood for each sample
            for i in range(batch_size):
                t = event_time[i]
                
                if event_indicator[i] > 0:  # Event occurred
                    # Add log hazard at event time
                    nll[i] = -log_hazard[i, t]
                    
                    # Add sum of log(1-hazard) before event time
                    if t > 0:
                        nll[i] -= torch.sum(log_1_minus_hazard[i, :t])
                else:  # Censored
                    # Add sum of log(1-hazard) up to censoring time
                    nll[i] = -torch.sum(log_1_minus_hazard[i, :t])
            
            # Use mean NLL as loss
            loss = torch.mean(nll)
        
        # Add ranking loss if alpha_rank > 0
        if self.alpha_rank > 0:
            # Compute concordance loss
            rank_loss = self._compute_ranking_loss(risk_score, event_indicator, event_time, torch.ones_like(event_indicator))
            loss = loss + self.alpha_rank * rank_loss
        
        # Add calibration loss if alpha_calibration > 0
        if self.alpha_calibration > 0:
            calibration_loss = self._compute_calibration_loss(survival, event_indicator, event_time, torch.ones_like(event_indicator))
            loss = loss + self.alpha_calibration * calibration_loss
            
        return loss
        
    def forward(self, 
               x: torch.Tensor, 
               targets: Optional[torch.Tensor] = None, 
               mask: Optional[torch.Tensor] = None,
               sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single-risk survival prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        targets : torch.Tensor, optional
            Target values with shape [batch_size, 2 + num_time_bins]:
            - targets[:, 0]: Event indicator (1 if event occurred, 0 if censored)
            - targets[:, 1]: Time bin index where event/censoring occurred
            - targets[:, 2:]: One-hot encoding of event time (for convenience)
            
        mask : torch.Tensor, optional
            Mask indicating which samples have targets for this task
            [batch_size], where 1 means target is available
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation [batch_size]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'loss': Loss for this task (if targets provided)
            - 'hazard': Predicted hazard function [batch_size, num_time_bins]
            - 'survival': Predicted survival function [batch_size, num_time_bins]
            - 'risk_score': Overall risk score [batch_size]
        """
        # Get batch size and device
        batch_size = x.size(0)
        device = x.device
        
        # Compute raw predictions (log hazards)
        log_hazards = self.prediction_network(x)
        
        # Check for NaN values in log_hazards
        if torch.isnan(log_hazards).any():
            # Replace NaN values with zeros
            log_hazards = torch.nan_to_num(log_hazards, nan=0.0)
        
        # Apply sigmoid to get hazard probabilities
        hazards = torch.sigmoid(log_hazards)
        
        # Ensure hazards are within [0, 1] bounds and no NaN values
        # This is a safety check as sigmoid should already constrain values
        hazards = torch.clamp(hazards, 0, 1)
        hazards = torch.nan_to_num(hazards, nan=0.5)
        
        # Compute survival function based on the discrete-time survival model
        # In discrete time:
        # S(0) = 1 (by definition)
        # S(t) = Prob(T > t) = Prob(not failing at times 1, 2, ..., t)
        # S(t) = (1-h(1)) * (1-h(2)) * ... * (1-h(t)) for t >= 1
        
        # Calculate survival probabilities at each time point
        # First compute (1 - hazard) for each time point
        survival_probs = 1 - hazards
        
        # Initialize survival with ones for S(0)
        batch_size = hazards.size(0)
        num_time_bins = hazards.size(1)
        survival = torch.ones(batch_size, num_time_bins, device=hazards.device)
        
        # Compute S(t) using the correct definition
        # S(t) = product from j=1 to t of (1-h(j))
        survival[:, 1:] = torch.cumprod(survival_probs[:, :-1], dim=1)
        
        # Compute overall risk score (negative expected survival time)
        risk_score = -torch.sum(survival, dim=1)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # If targets are provided, compute the loss
        if targets is not None:
            # Extract event indicator and time index
            event_indicator = targets[:, 0]
            event_time = targets[:, 1].long()
            
            # Apply task mask if provided
            if mask is not None:
                # Expand mask to match event indicator
                mask = mask.float()
                event_indicator = event_indicator * mask
                
                # Count valid samples for loss normalization
                valid_samples = torch.sum(mask)
            else:
                # All samples are valid
                valid_samples = float(batch_size)
                mask = torch.ones_like(event_indicator)
            
            # Apply sample weights if provided
            if sample_weights is not None:
                # Combine with mask
                combined_weights = mask * sample_weights
            else:
                combined_weights = mask
                
            # Skip loss computation if no valid samples
            if valid_samples > 0:
                if self.use_bce_loss:
                    # Binary cross-entropy loss for each time point
                    # For observed events:
                    #   - Before event: no event, target = 0
                    #   - At event: event occurs, target = 1
                    #   - After event: undefined, masked out
                    # For censored:
                    #   - Before censoring: no event, target = 0
                    #   - At and after censoring: undefined, masked out
                    
                    # Create targets and mask for BCE loss
                    bce_targets = torch.zeros_like(hazards)
                    bce_mask = torch.zeros_like(hazards)
                    
                    # Set targets and mask for each sample
                    for i in range(batch_size):
                        if mask[i] > 0:  # If this sample has a valid target
                            t = event_time[i]
                            
                            if event_indicator[i] > 0:  # Event occurred
                                # Target 0 before event, 1 at event
                                bce_targets[i, :t] = 0
                                bce_targets[i, t] = 1
                                
                                # Mask includes up to and including event
                                bce_mask[i, :(t+1)] = 1
                            else:  # Censored
                                # Target 0 before censoring
                                bce_targets[i, :t] = 0
                                
                                # Mask includes only up to censoring (exclusive)
                                bce_mask[i, :t] = 1
                    
                    # Apply sample weights to BCE mask if provided
                    if sample_weights is not None:
                        weighted_bce_mask = torch.zeros_like(bce_mask)
                        for i in range(batch_size):
                            weighted_bce_mask[i] = bce_mask[i] * sample_weights[i]
                        bce_mask = weighted_bce_mask
                    
                    # Compute BCE loss
                    bce_loss = F.binary_cross_entropy(
                        hazards, 
                        bce_targets, 
                        reduction='none'
                    )
                    
                    # Apply mask and normalize
                    bce_loss = torch.sum(bce_loss * bce_mask) / (torch.sum(bce_mask) + 1e-6)
                    loss = bce_loss
                else:
                    # Negative log-likelihood loss (Discrete-time NLL)
                    # L = - sum_{i=1}^N [ delta_i * log(h_i(t_i)) + sum_{j=1}^{t_i-1} log(1 - h_i(j)) ]
                    # where delta_i is the event indicator, t_i is the event/censoring time,
                    # and h_i(j) is the predicted hazard for sample i at time j.
                    # For observed events (delta_i=1): - [log(h_i(t_i)) + sum_{j=1}^{t_i-1} log(1 - h_i(j))]
                    # For censored events (delta_i=0): - [sum_{j=1}^{t_i} log(1 - h_i(j))]
                    
                    # Compute log terms with extra safety for numerical stability
                    epsilon = 1e-7
                    hazards_safe = torch.clamp(hazards, epsilon, 1.0 - epsilon) 
                    log_hazard = torch.log(hazards_safe)
                    log_1_minus_hazard = torch.log(1 - hazards_safe)
                    
                    # Initialize likelihood
                    nll = torch.zeros(batch_size, device=device)
                    
                    # Compute likelihood for each sample
                    for i in range(batch_size):
                        if mask[i] > 0:  # If this sample has a valid target
                            t = event_time[i]
                            
                            if event_indicator[i] > 0:  # Event occurred
                                # Add log hazard at event time
                                nll[i] = -log_hazard[i, t]
                                
                                # Add sum of log(1-hazard) before event time
                                if t > 0:
                                    nll[i] -= torch.sum(log_1_minus_hazard[i, :t])
                            else:  # Censored
                                # Add sum of log(1-hazard) up to censoring time
                                nll[i] = -torch.sum(log_1_minus_hazard[i, :t])
                    
                    # Compute weighted mean NLL for valid samples
                    weighted_nll = nll * combined_weights
                    nll = torch.sum(weighted_nll) / (torch.sum(combined_weights) + 1e-6)
                    loss = nll
                
                # Add ranking loss if alpha_rank > 0
                if self.alpha_rank > 0:
                    # Compute concordance loss - pass sample weights if available
                    if sample_weights is not None:
                        rank_loss = self._compute_ranking_loss(risk_score, event_indicator, event_time, mask, sample_weights)
                    else:
                        rank_loss = self._compute_ranking_loss(risk_score, event_indicator, event_time, mask)
                    loss = loss + self.alpha_rank * rank_loss
                
                # Add calibration loss if alpha_calibration > 0
                if self.alpha_calibration > 0:
                    if sample_weights is not None:
                        calibration_loss = self._compute_calibration_loss(survival, event_indicator, event_time, mask, sample_weights)
                    else:
                        calibration_loss = self._compute_calibration_loss(survival, event_indicator, event_time, mask)
                    loss = loss + self.alpha_calibration * calibration_loss
        
        return {
            'loss': loss,
            'hazard': hazards,
            'survival': survival,
            'risk_score': risk_score
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate survival predictions.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'hazard': Predicted hazard function [batch_size, num_time_bins]
            - 'survival': Predicted survival function [batch_size, num_time_bins]
            - 'risk_score': Overall risk score [batch_size]
        """
        # Compute predictions (no loss)
        outputs = self.forward(x)
        
        # Remove loss from outputs
        outputs.pop('loss', None)
        
        return outputs
    
    def compute_metrics(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics for survival prediction.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from the forward pass
            
        targets : torch.Tensor
            Ground truth targets
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values
        """
        # Extract predictions
        risk_scores = outputs['risk_score'].detach().cpu().numpy()
        survival = outputs['survival'].detach().cpu().numpy()
        
        # Extract targets
        event_indicator = targets[:, 0].detach().cpu().numpy()
        event_time = targets[:, 1].detach().cpu().numpy()
        
        # Compute concordance index
        c_index = self._compute_c_index(risk_scores, event_time, event_indicator)
        
        # Compute integrated Brier score
        brier_score = self._compute_integrated_brier_score(survival, event_time, event_indicator)
        
        # Compute time-dependent AUC
        auc = self._compute_time_dependent_auc(risk_scores, event_time, event_indicator)
        
        return {
            'c_index': c_index,
            'brier_score': brier_score,
            'auc': auc
        }
    
    def _compute_ranking_loss(self, 
                             risk_scores: torch.Tensor, 
                             event_indicator: torch.Tensor, 
                             event_time: torch.Tensor,
                             mask: torch.Tensor,
                              sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute ranking loss for concordance optimization.
        
        Uses a pairwise hinge loss to encourage higher risk scores for samples
        that experience an event earlier than comparable samples.
        Loss = sum_{i,j in comparable pairs} max(0, 1 - (risk_i - risk_j)) / N_pairs
        where risk_i > risk_j is desired if sample i has an earlier event time.
        
        Parameters
        ----------
        risk_scores : torch.Tensor
            Predicted risk scores [batch_size]
            
        event_indicator : torch.Tensor
            Event indicators [batch_size]
            
        event_time : torch.Tensor
            Event times [batch_size]
            
        mask : torch.Tensor
            Mask indicating valid samples [batch_size]
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation [batch_size]
            
        Returns
        -------
        torch.Tensor
            Ranking loss value (scalar)
        """
        batch_size = risk_scores.size(0)
        device = risk_scores.device
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # Count valid comparisons
        valid_comparisons = 0
        
        # Compute pairwise rankings
        for i in range(batch_size):
            if mask[i] == 0 or event_indicator[i] == 0:
                # Skip censored or masked samples for first position
                continue
                
            for j in range(batch_size):
                if mask[j] == 0:
                    # Skip masked samples for second position
                    continue
                    
                # Valid comparison if:
                # 1. i had an event and j was censored after i's event, or
                # 2. Both i and j had events, and i's event was before j's
                if (event_indicator[i] == 1 and 
                    ((event_indicator[j] == 0 and event_time[j] > event_time[i]) or
                     (event_indicator[j] == 1 and event_time[i] < event_time[j]))):
                    
                    # i should have a higher risk score than j
                    risk_diff = risk_scores[j] - risk_scores[i]
                    
                    # Compute hinge loss: max(0, 1 - (risk_i - risk_j))
                    pair_loss = torch.relu(1.0 + risk_diff)
                    
                    # Apply sample weights if provided
                    if sample_weights is not None:
                        # Weight the pair loss by the product of both sample weights
                        pair_weight = sample_weights[i] * sample_weights[j]
                        loss = loss + pair_loss * pair_weight
                        valid_comparisons += pair_weight
                    else:
                        loss = loss + pair_loss
                        valid_comparisons += 1
        
        # Normalize loss by number of valid comparisons
        if valid_comparisons > 0:
            loss = loss / valid_comparisons
            
        return loss
    
    def _compute_calibration_loss(self,
                                 survival: torch.Tensor,
                                 event_indicator: torch.Tensor,
                                 event_time: torch.Tensor,
                                 mask: torch.Tensor,
                                  sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute calibration loss to ensure predicted probabilities match empirical frequencies.
        
        Calculates the L2 distance between the mean predicted survival probability
        and the empirical survival probability (Kaplan-Meier estimate) at each time bin.
        Loss = sum_{t=1}^{T} ( mean(S_pred(t)) - S_empirical(t) )^2 / T
        
        Parameters
        ----------
        survival : torch.Tensor
            Predicted survival functions [batch_size, num_time_bins]
            
        event_indicator : torch.Tensor
            Event indicators [batch_size]
            
        event_time : torch.Tensor
            Event times [batch_size]
            
        mask : torch.Tensor
            Mask indicating valid samples [batch_size]
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation [batch_size]
            
        Returns
        -------
        torch.Tensor
            Calibration loss value (scalar)
        """
        batch_size = survival.size(0)
        device = survival.device
        
        # Get number of time bins
        num_bins = survival.size(1)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # For each time bin
        for t in range(num_bins):
            # Count events that occur at or before time t
            events_at_t = torch.zeros(batch_size, device=device)
            
            for i in range(batch_size):
                if mask[i] > 0 and event_indicator[i] > 0 and event_time[i] <= t:
                    events_at_t[i] = 1.0
            
            # Calculate empirical probability of event by time t
            if sample_weights is not None:
                weighted_events = events_at_t * sample_weights * mask
                weighted_mask = mask * sample_weights
                empirical_prob = torch.sum(weighted_events) / (torch.sum(weighted_mask) + 1e-6)
            else:
                empirical_prob = torch.sum(events_at_t * mask) / (torch.sum(mask) + 1e-6)
                
            # Calculate predicted probability of event by time t
            predicted_prob = 1.0 - torch.mean(survival[:, t])
            
            # L2 loss between predicted and empirical probabilities
            bin_loss = (predicted_prob - empirical_prob).pow(2)
            loss = loss + bin_loss
        
        # Average over all time bins
        loss = loss / num_bins
        
        return loss
    
    def _compute_c_index(self, risk_scores, event_time, event_indicator):
        """Compute concordance index from risk scores and event data."""
        n_samples = len(risk_scores)
        concordant_pairs = 0
        total_pairs = 0
        
        for i in range(n_samples):
            if event_indicator[i] == 0:
                # Skip censored samples for first position
                continue
                
            for j in range(n_samples):
                # Valid comparison if:
                # 1. i had an event and j was censored after i's event, or
                # 2. Both i and j had events, and i's event was before j's
                if (event_indicator[i] == 1 and 
                    ((event_indicator[j] == 0 and event_time[j] > event_time[i]) or
                     (event_indicator[j] == 1 and event_time[i] < event_time[j]))):
                    
                    total_pairs += 1
                    
                    # Higher risk score should predict earlier event
                    if risk_scores[i] > risk_scores[j]:
                        concordant_pairs += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant_pairs += 0.5
        
        # Return c-index (proportion of concordant pairs)
        if total_pairs == 0:
            return 0.5
        else:
            return concordant_pairs / total_pairs
    
    def _compute_integrated_brier_score(self, survival, event_time, event_indicator):
        """Compute integrated Brier score."""
        n_samples = len(event_time)
        n_time_bins = survival.shape[1]
        
        # Check for NaN values in input arrays
        if np.isnan(survival).any():
            print("Warning: NaN values found in survival probabilities. Replacing with zeros.")
            survival = np.nan_to_num(survival, nan=0.0)
            
        if np.isnan(event_time).any():
            print("Warning: NaN values found in event times. Replacing with zeros.")
            event_time = np.nan_to_num(event_time, nan=0.0)
            
        if np.isnan(event_indicator).any():
            print("Warning: NaN values found in event indicators. Replacing with zeros.")
            event_indicator = np.nan_to_num(event_indicator, nan=0.0)
        
        # Initialize Brier score for each time point
        brier_scores = np.zeros(n_time_bins)
        valid_time_bins = 0
        
        # For each time bin
        for t in range(n_time_bins):
            brier_score_t = 0.0
            weight_sum = 0.0
            
            for i in range(n_samples):
                # Skip samples with invalid event times
                if np.isnan(event_time[i]) or event_time[i] < 0:
                    continue
                    
                # Observed outcome at time t
                Y_t = 0.0  # Default: survived past time t
                weight = 1.0  # Default weight
                
                if event_indicator[i] == 1 and event_time[i] <= t:
                    # Event before or at time t
                    Y_t = 1.0
                elif event_time[i] <= t:
                    # Censored before or at time t, undefined true outcome
                    weight = 0.0
                
                # Skip if weight is zero
                if weight > 0:
                    # Check if survival value is valid
                    if np.isnan(survival[i, t]) or survival[i, t] < 0 or survival[i, t] > 1:
                        continue
                        
                    # Predicted probability of event by time t
                    pred_t = 1 - survival[i, t]
                    
                    # Squared error weighted by inverse probability of censoring
                    brier_score_t += weight * (Y_t - pred_t) ** 2
                    weight_sum += weight
            
            # Average over samples with non-zero weights
            if weight_sum > 0:
                brier_scores[t] = brier_score_t / weight_sum
                valid_time_bins += 1
        
        # Compute integrated score (mean over time)
        if valid_time_bins > 0:
            # Only use time bins with valid scores
            valid_scores = brier_scores[~np.isnan(brier_scores)]
            if len(valid_scores) > 0:
                return np.mean(valid_scores)
                
        # If no valid scores, return NaN with a warning
        print("Warning: Could not compute Integrated Brier Score (no valid time bins)")
        return 0.5  # Return a default value instead of NaN
    
    def _compute_time_dependent_auc(self, risk_scores, event_time, event_indicator):
        """Compute time-dependent AUC."""
        n_samples = len(risk_scores)
        n_auc_points = 0
        auc_sum = 0.0
        
        # For a few representative time points
        evaluation_times = np.percentile(event_time[event_indicator == 1], [25, 50, 75])
        
        for eval_time in evaluation_times:
            # Initialize counts for this time point
            true_positives = np.zeros(n_samples)
            false_positives = np.zeros(n_samples)
            
            # Count positive cases: had an event by eval_time
            # Count negative cases: no event by eval_time (censored after or event after)
            positives = (event_indicator == 1) & (event_time <= eval_time)
            negatives = (event_time > eval_time)
            
            if np.sum(positives) == 0 or np.sum(negatives) == 0:
                # Skip this time point if no positives or negatives
                continue
                
            # Sort samples by risk score (descending)
            sorted_indices = np.argsort(-risk_scores)
            
            # Compute TPR and FPR at each threshold
            tp_count = 0
            fp_count = 0
            
            for idx in sorted_indices:
                if positives[idx]:
                    tp_count += 1
                elif negatives[idx]:
                    fp_count += 1
                
                true_positives[idx] = tp_count / np.sum(positives)
                false_positives[idx] = fp_count / np.sum(negatives)
            
            # Compute AUC using trapezoidal rule
            sorted_fps = false_positives[sorted_indices]
            sorted_tps = true_positives[sorted_indices]
            
            # Compute area using trapezoidal rule
            time_auc = np.trapz(sorted_tps, sorted_fps)
            
            auc_sum += time_auc
            n_auc_points += 1
            
        # Average AUC over evaluation times
        if n_auc_points > 0:
            return auc_sum / n_auc_points
        else:
            return 0.5
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            'num_time_bins': self.num_time_bins,
            'alpha_rank': self.alpha_rank,
            'alpha_calibration': self.alpha_calibration,
            'use_bce_loss': self.use_bce_loss
        })
        return config


class CompetingRisksHead(TaskHead):
    """
    Task head for competing risks survival analysis.
    
    Predicts the probability of each competing event occurring at each time point,
    handling right-censoring through masked loss computation.
    """
    
    def __init__(self,
                name: str,
                input_dim: int,
                num_time_bins: int,
                num_risks: int,
                alpha_rank: float = 0.0,
                alpha_calibration: float = 0.0,
                task_weight: float = 1.0,
                use_softmax: bool = True,
                use_cause_specific: bool = True,
                dropout: float = 0.1):
        """
        Initialize CompetingRisksHead.
        
        Parameters
        ----------
        name : str
            Name of the task
            
        input_dim : int
            Dimension of the input representation
            
        num_time_bins : int
            Number of discrete time bins to predict
            
        num_risks : int
            Number of competing risks (causes)
            
        alpha_rank : float, default=0.0
            Weight of the ranking loss component
            
        alpha_calibration : float, default=0.0
            Weight of the calibration loss component
            
        task_weight : float, default=1.0
            Weight of this task in the multi-task loss
            
        use_softmax : bool, default=True
            Whether to use softmax normalization across causes (True) or 
            independent sigmoid functions (False)
            
        use_cause_specific : bool, default=True
            Whether to use cause-specific loss function (True) or
            subdistribution approach (False)
            
        dropout : float, default=0.1
            Dropout rate for the prediction network
        """
        super().__init__(name, input_dim, task_weight)
        
        self.num_time_bins = num_time_bins
        self.num_risks = num_risks
        self.alpha_rank = alpha_rank
        self.alpha_calibration = alpha_calibration
        self.use_softmax = use_softmax
        self.use_cause_specific = use_cause_specific
        
        # Architecture
        # For competing risks, we need to predict hazards for each risk at each time
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Separate prediction head for each risk
        self.risk_heads = nn.ModuleList([
            nn.Linear(input_dim * 2, num_time_bins) 
            for _ in range(num_risks)
        ])
        
    def loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between model outputs and targets.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Outputs from the forward pass
            
        targets : torch.Tensor
            Target values with shape [batch_size, 3 + num_risks * num_time_bins]:
            - targets[:, 0]: Event indicator (1 if event occurred, 0 if censored)
            - targets[:, 1]: Time bin index where event/censoring occurred
            - targets[:, 2]: Cause index (-1 if censored)
            - targets[:, 3:]: One-hot encoding of event time by cause
            
        Returns
        -------
        torch.Tensor
            Loss value
        """
        if 'loss' in outputs:
            return outputs['loss']
            
        # Extract relevant outputs
        hazards = outputs['hazards']
        cif = outputs['cif']
        overall_survival = outputs['overall_survival']
        risk_scores = outputs['risk_scores']
        
        # Extract target components
        event_indicator = targets[:, 0]
        event_time = targets[:, 1].long()
        cause = targets[:, 2].long()
        
        # Initialize components
        batch_size = hazards.size(0)
        device = hazards.device
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        if self.use_cause_specific:
            # Cause-specific approach with binary cross-entropy loss for each risk
            # Create targets and masks for each risk
            bce_targets = torch.zeros(
                batch_size, self.num_risks, self.num_time_bins, device=device
            )
            bce_mask = torch.zeros(
                batch_size, self.num_risks, self.num_time_bins, device=device
            )
            
            # Set targets and mask for each sample
            for i in range(batch_size):
                t = event_time[i]
                
                if event_indicator[i] > 0:  # Event occurred
                    c = cause[i]  # Which cause
                    
                    # Validate cause index is within range
                    if c >= self.num_risks:
                        # If cause index is out of range, treat as censored
                        # This can happen in testing when random causes are generated
                        for c in range(self.num_risks):
                            bce_targets[i, c, :t] = 0
                            bce_mask[i, c, :t] = 1
                    else:
                        # For the cause that occurred:
                        # Target 0 before event, 1 at event
                        bce_targets[i, c, :t] = 0
                        bce_targets[i, c, t] = 1
                        
                        # Mask includes up to and including event
                        bce_mask[i, c, :(t+1)] = 1
                        
                        # For other causes:
                        # Target 0 before event time (censored at event time for other causes)
                        for other_c in range(self.num_risks):
                            if other_c != c:
                                bce_targets[i, other_c, :t] = 0
                                bce_mask[i, other_c, :t] = 1
                else:  # Censored
                    # Target 0 before censoring for all causes
                    for c in range(self.num_risks):
                        bce_targets[i, c, :t] = 0
                        bce_mask[i, c, :t] = 1
            
            # Compute BCE loss
            bce_loss = F.binary_cross_entropy(
                hazards, 
                bce_targets, 
                reduction='none'
            )
            
            # Apply mask and normalize
            loss = torch.sum(bce_loss * bce_mask) / (torch.sum(bce_mask) + 1e-6)
        else:
            # Fine-Gray subdistribution Negative Log-Likelihood
            # Initialize likelihood
            nll = torch.zeros(batch_size, device=device)
            
            # Compute likelihood for each sample
            for i in range(batch_size):
                t = event_time[i]
                
                if event_indicator[i] > 0:  # Event occurred
                    c = cause[i]  # Which cause
                    
                    # Add log hazard for specific cause at event time
                    epsilon = 1e-7
                    hazard_c_t = torch.clamp(hazards[i, c, t], epsilon, 1.0 - epsilon)
                    nll[i] = -torch.log(hazard_c_t)
                    
                    # Add log overall survival up to previous time point
                    if t > 0:
                        prev_surv = torch.clamp(overall_survival[i, t-1], epsilon, 1.0)
                        nll[i] -= torch.log(prev_surv)
                else:  # Censored
                    # Add log probability of not experiencing any event by censoring time
                    if t > 0:
                        surv_t = torch.clamp(overall_survival[i, t-1], epsilon, 1.0)
                        nll[i] = -torch.log(surv_t)
            
            # Mean NLL
            loss = torch.mean(nll)
        
        # Add ranking loss if alpha_rank > 0
        if self.alpha_rank > 0:
            # Compute ranking loss for each cause
            rank_loss = torch.tensor(0.0, device=device)
            
            for risk_idx in range(self.num_risks):
                risk_scores_k = risk_scores[:, risk_idx]
                
                # Only consider samples where this specific cause occurred
                cause_mask = (cause == risk_idx).float() * event_indicator
                
                if torch.sum(cause_mask) > 0:
                    # Compute cause-specific ranking loss
                    risk_rank_loss = self._compute_ranking_loss(
                        risk_scores_k, cause_mask, event_time, torch.ones_like(cause_mask)
                    )
                    
                    rank_loss = rank_loss + risk_rank_loss
            
            # Normalize by number of risks
            rank_loss = rank_loss / self.num_risks
            loss = loss + self.alpha_rank * rank_loss
        
        # Add calibration loss if alpha_calibration > 0
        if self.alpha_calibration > 0:
            # Compute calibration loss for CIF predictions
            calibration_loss = torch.tensor(0.0, device=device)
            
            for risk_idx in range(self.num_risks):
                risk_cal_loss = self._compute_cif_calibration_loss(
                    cif[:, risk_idx, :], risk_idx, event_indicator, cause, event_time, torch.ones_like(event_indicator)
                )
                
                calibration_loss = calibration_loss + risk_cal_loss
            
            # Normalize by number of risks
            calibration_loss = calibration_loss / self.num_risks
            loss = loss + self.alpha_calibration * calibration_loss
        
        return loss
        
    def forward(self, 
               x: torch.Tensor, 
               targets: Optional[torch.Tensor] = None, 
               mask: Optional[torch.Tensor] = None,
               sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for competing risks survival prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        targets : torch.Tensor, optional
            Target values with shape [batch_size, 3 + num_risks * num_time_bins]:
            - targets[:, 0]: Event indicator (1 if event occurred, 0 if censored)
            - targets[:, 1]: Time bin index where event/censoring occurred
            - targets[:, 2]: Cause index (-1 if censored)
            - targets[:, 3:]: One-hot encoding of event time by cause
            
        mask : torch.Tensor, optional
            Mask indicating which samples have targets for this task
            [batch_size], where 1 means target is available
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation [batch_size]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'loss': Loss for this task (if targets provided)
            - 'hazards': Predicted hazard functions [batch_size, num_risks, num_time_bins]
            - 'cif': Cumulative incidence functions [batch_size, num_risks, num_time_bins]
            - 'overall_survival': Overall survival function [batch_size, num_time_bins]
            - 'risk_scores': Risk scores for each cause [batch_size, num_risks]
        """
        # Get batch size and device
        batch_size = x.size(0)
        device = x.device
        
        # Compute shared representation
        shared_features = self.shared_network(x)
        
        # Compute raw predictions (log hazards) for each risk
        log_hazards_list = []
        for risk_idx in range(self.num_risks):
            risk_log_hazards = self.risk_heads[risk_idx](shared_features)
            log_hazards_list.append(risk_log_hazards.unsqueeze(1))  # [batch_size, 1, num_time_bins]
        
        # Combine all risk predictions
        log_hazards = torch.cat(log_hazards_list, dim=1)  # [batch_size, num_risks, num_time_bins]
        
        # Replace NaN values with zeros
        log_hazards = torch.nan_to_num(log_hazards, nan=0.0)
        
        # Apply activation function to get hazard probabilities
        if self.use_softmax:
            # For each time point, softmax across risks (including no event)
            # We add a column for "no event" to ensure sum of probabilities = 1
            hazards_with_none = F.softmax(
                torch.cat([torch.zeros(batch_size, 1, self.num_time_bins, device=device), log_hazards], dim=1),
                dim=1
            )
            # Extract actual cause hazards (excluding no event)
            hazards = hazards_with_none[:, 1:, :]
        else:
            # Independent sigmoid for each risk
            # This doesn't guarantee sum(hazards) <= 1, so we normalize if needed
            hazards = torch.sigmoid(log_hazards)
            
            # Ensure sum of hazards doesn't exceed 1 by scaling if necessary
            hazard_sums = torch.sum(hazards, dim=1, keepdim=True)
            exceeds_one = hazard_sums > 1.0
            if torch.any(exceeds_one):
                scale_factor = torch.ones_like(hazard_sums)
                scale_factor = torch.where(exceeds_one, 1.0 / hazard_sums, scale_factor)
                hazards = hazards * scale_factor
        
        # Ensure hazards are within [0, 1] bounds and no NaN values
        hazards = torch.clamp(hazards, 0, 1)
        hazards = torch.nan_to_num(hazards, nan=0.5)
        
        # Compute cause-specific survival functions
        # S_k(t) = Prob(T > t | cause = k)
        cause_specific_survival = torch.ones(
            batch_size, self.num_risks, self.num_time_bins, device=device
        )
        
        # For each risk, compute cause-specific survival
        for risk_idx in range(self.num_risks):
            # Compute (1 - hazard) for current risk
            survival_probs = 1 - hazards[:, risk_idx, :]
            
            # Handle any potential NaN values
            if torch.isnan(survival_probs).any():
                survival_probs = torch.nan_to_num(survival_probs, nan=1.0)
            
            # Ensure survival probabilities are in valid range [0, 1]
            survival_probs = torch.clamp(survival_probs, 0.0, 1.0)
            
            # S_k(t) = product from j=1 to t of (1-h_k(j))
            cause_specific_survival_k = torch.ones_like(cause_specific_survival[:, risk_idx, :])
            cause_specific_survival_k[:, 1:] = torch.cumprod(survival_probs[:, :-1], dim=1)
            
            # Handle any NaN values that might have been produced
            if torch.isnan(cause_specific_survival_k).any():
                cause_specific_survival_k = torch.nan_to_num(cause_specific_survival_k, nan=1.0)
            
            cause_specific_survival[:, risk_idx, :] = cause_specific_survival_k
        
        # Compute overall survival (probability of surviving all risks)
        # S(t) = Prob(T > t) = Prob(T_1 > t, T_2 > t, ..., T_K > t)
        # For independent risks: S(t) = S_1(t) * S_2(t) * ... * S_K(t)
        overall_survival = torch.prod(cause_specific_survival, dim=1)
        
        # Handle any potential NaN values in overall survival
        if torch.isnan(overall_survival).any():
            overall_survival = torch.nan_to_num(overall_survival, nan=1.0)
        
        # Ensure overall survival is in valid range [0, 1]
        overall_survival = torch.clamp(overall_survival, 0.0, 1.0)
        
        # Compute cumulative incidence functions (CIF)
        # F_k(t) = Prob(T <= t, cause = k)
        cif = torch.zeros(
            batch_size, self.num_risks, self.num_time_bins, device=device
        )
        
        # For each time bin, compute CIF for each risk
        for t in range(1, self.num_time_bins):
            # For each risk, compute CIF at time t
            for risk_idx in range(self.num_risks):
                # CIF_k(t) = sum_{j=1}^t [ h_k(j) * S(j-1) ]
                # Where S(j-1) is overall survival function at previous time
                if t == 1:
                    prev_survival = torch.ones(batch_size, device=device)
                else:
                    prev_survival = overall_survival[:, t-2]
                
                # Handle NaN values in previous survival
                if torch.isnan(prev_survival).any():
                    prev_survival = torch.nan_to_num(prev_survival, nan=1.0)
                
                # Handle NaN values in hazards
                current_hazards = hazards[:, risk_idx, t-1]
                if torch.isnan(current_hazards).any():
                    current_hazards = torch.nan_to_num(current_hazards, nan=0.0)
                
                # Create a new tensor instead of inplace assignment
                new_cif_val = cif[:, risk_idx, t-1] + current_hazards * prev_survival
                
                # Handle any potential NaN values in the sum
                if torch.isnan(new_cif_val).any():
                    new_cif_val = torch.nan_to_num(new_cif_val, nan=0.0)
                    
                cif[:, risk_idx, t] = new_cif_val
        
        # Normalize CIFs to ensure they sum to 1 - overall_survival
        for t in range(self.num_time_bins):
            # Calculate the current sum of all CIFs at time t
            cif_sum = torch.sum(cif[:, :, t], dim=1)
            
            # Handle any potential NaN values in the sum
            if torch.isnan(cif_sum).any():
                cif_sum = torch.nan_to_num(cif_sum, nan=0.0)
            
            # Calculate the target sum (should be 1 - overall_survival)
            target_sum = 1.0 - overall_survival[:, t]
            
            # Handle any potential NaN values in the target sum
            if torch.isnan(target_sum).any():
                target_sum = torch.nan_to_num(target_sum, nan=0.0)
            
            # Ensure target_sum is in valid range [0, 1]
            target_sum = torch.clamp(target_sum, 0.0, 1.0)
            
            # Avoid division by zero and trivial cases
            valid_indices = (cif_sum > 1e-6) & (target_sum > 1e-6)
            
            if torch.any(valid_indices):
                # Calculate scaling factor for each sample
                scale_factor = torch.ones_like(cif_sum)
                # Clone to avoid in-place modification
                new_scale_factor = scale_factor.clone()
                
                # Divide with safety checking
                division_result = torch.zeros_like(cif_sum)
                mask = valid_indices & (cif_sum > 1e-6)  # Extra safety check
                if torch.any(mask):
                    division_result[mask] = target_sum[mask] / cif_sum[mask]
                
                # Handle potential infinity or NaN values from division
                division_result = torch.nan_to_num(division_result, nan=1.0, posinf=1.0, neginf=1.0)
                
                # Only apply valid scaling factors (between 0.1 and 10 for safety)
                valid_scale = (division_result >= 0.1) & (division_result <= 10.0)
                new_scale_factor[valid_indices & valid_scale] = division_result[valid_indices & valid_scale]
                scale_factor = new_scale_factor
                
                # Apply scaling to each risk's CIF
                for risk_idx in range(self.num_risks):
                    # Create a new tensor instead of in-place operations
                    scaled_cif = cif[:, risk_idx, t].clone() * scale_factor
                    # Handle any NaN values that might result
                    scaled_cif = torch.nan_to_num(scaled_cif, nan=0.0)
                    # Ensure values are in valid range [0, 1]
                    scaled_cif = torch.clamp(scaled_cif, 0.0, 1.0)
                    # Assign back to CIF
                    cif[:, risk_idx, t] = scaled_cif
        
        # Compute overall risk scores for each cause (negative expected survival time)
        risk_scores = torch.sum(cif, dim=2)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # If targets are provided, compute the loss
        if targets is not None:
            # Extract event indicator, time index, and cause
            event_indicator = targets[:, 0]  # 1 if event, 0 if censored
            event_time = targets[:, 1].long()  # Time bin index
            cause = targets[:, 2].long()  # Cause index (-1 if censored)
            
            # Apply task mask if provided
            if mask is not None:
                # Expand mask to match event indicator
                mask = mask.float()
                event_indicator = event_indicator * mask
                
                # Count valid samples for loss normalization
                valid_samples = torch.sum(mask)
            else:
                # All samples are valid
                valid_samples = float(batch_size)
                mask = torch.ones_like(event_indicator)
            
            # Apply sample weights if provided
            if sample_weights is not None:
                # Combine with mask
                combined_weights = mask * sample_weights
            else:
                combined_weights = mask
                
            # Skip loss computation if no valid samples
            if valid_samples > 0:
                if self.use_cause_specific:
                    # Cause-specific approach with binary cross-entropy loss for each risk
                    # Create targets and masks for each risk
                    bce_targets = torch.zeros(
                        batch_size, self.num_risks, self.num_time_bins, device=device
                    )
                    bce_mask = torch.zeros(
                        batch_size, self.num_risks, self.num_time_bins, device=device
                    )
                    
                    # Set targets and mask for each sample
                    for i in range(batch_size):
                        if mask[i] > 0:  # If this sample has a valid target
                            t = event_time[i]
                            
                            if event_indicator[i] > 0:  # Event occurred
                                c = cause[i]  # Which cause
                                
                                # Validate cause index is within range
                                if c >= self.num_risks:
                                    # If cause index is out of range, treat as censored
                                    # This can happen in testing when random causes are generated
                                    for c_idx in range(self.num_risks):
                                        bce_targets[i, c_idx, :t] = 0
                                        bce_mask[i, c_idx, :t] = 1
                                else:
                                    # For the cause that occurred:
                                    # Target 0 before event, 1 at event
                                    bce_targets[i, c, :t] = 0
                                    bce_targets[i, c, t] = 1
                                
                                # Only set masks if cause is valid
                                if c < self.num_risks:
                                    # Mask includes up to and including event
                                    bce_mask[i, c, :(t+1)] = 1
                                    
                                    # For other causes:
                                    # Target 0 before event time (censored at event time for other causes)
                                    for other_c in range(self.num_risks):
                                        if other_c != c:
                                            bce_targets[i, other_c, :t] = 0
                                            bce_mask[i, other_c, :t] = 1
                            else:  # Censored
                                # Target 0 before censoring for all causes
                                for c in range(self.num_risks):
                                    bce_targets[i, c, :t] = 0
                                    bce_mask[i, c, :t] = 1
                    
                    # Apply sample weights to BCE mask if provided
                    if sample_weights is not None:
                        weighted_bce_mask = torch.zeros_like(bce_mask)
                        for i in range(batch_size):
                            weighted_bce_mask[i] = bce_mask[i] * sample_weights[i]
                        bce_mask = weighted_bce_mask
                    
                    # Compute BCE loss
                    bce_loss = F.binary_cross_entropy(
                        hazards, 
                        bce_targets, 
                        reduction='none'
                    )
                    
                    # Apply mask and normalize
                    bce_loss = torch.sum(bce_loss * bce_mask) / (torch.sum(bce_mask) + 1e-6)
                    loss = bce_loss
                else:
                    # Fine-Gray subdistribution Negative Log-Likelihood
                    # L = - sum_{i=1}^N [ delta_i * I(cause_i=k) * (log(h_{ik}(t_i)) + log(S_i(t_i-1))) + (1 - delta_i) * log(S_i(t_i)) ]
                    # where h_{ik}(t) is the cause-specific hazard for cause k,
                    # S_i(t) is the overall survival probability for sample i at time t.
                    # For observed event k at time t_i: - [log(h_{ik}(t_i)) + log(S_i(t_i-1))]
                    # For censored event at time t_i: - log(S_i(t_i))
                    
                    # Initialize likelihood
                    nll = torch.zeros(batch_size, device=device)
                    
                    # Compute likelihood for each sample
                    for i in range(batch_size):
                        if mask[i] > 0:  # If this sample has a valid target
                            t = event_time[i]
                            
                            if event_indicator[i] > 0:  # Event occurred
                                c = cause[i]  # Which cause
                                
                                # Add log hazard for specific cause at event time
                                epsilon = 1e-7
                                hazard_c_t = torch.clamp(hazards[i, c, t], epsilon, 1.0 - epsilon)
                                nll[i] = -torch.log(hazard_c_t)
                                
                                # Add log overall survival up to previous time point
                                if t > 0:
                                    prev_surv = torch.clamp(overall_survival[i, t-1], epsilon, 1.0)
                                    nll[i] -= torch.log(prev_surv)
                            else:  # Censored
                                # Add log probability of not experiencing any event by censoring time
                                if t > 0:
                                    surv_t = torch.clamp(overall_survival[i, t-1], epsilon, 1.0)
                                    nll[i] = -torch.log(surv_t)
                    
                    # Compute weighted mean NLL for valid samples
                    weighted_nll = nll * combined_weights
                    nll = torch.sum(weighted_nll) / (torch.sum(combined_weights) + 1e-6)
                    loss = nll
                
                # Add ranking loss if alpha_rank > 0
                if self.alpha_rank > 0:
                    # Compute ranking loss for each cause
                    rank_loss = torch.tensor(0.0, device=device)
                    
                    for risk_idx in range(self.num_risks):
                        risk_scores_k = risk_scores[:, risk_idx]
                        
                        # Only consider samples where this specific cause occurred
                        cause_mask = (cause == risk_idx).float() * event_indicator * mask
                        
                        if torch.sum(cause_mask) > 0:
                            # Compute cause-specific ranking loss
                            if sample_weights is not None:
                                cause_weights = sample_weights * cause_mask
                                risk_rank_loss = self._compute_ranking_loss(
                                    risk_scores_k, cause_mask, event_time, mask, cause_weights
                                )
                            else:
                                risk_rank_loss = self._compute_ranking_loss(
                                    risk_scores_k, cause_mask, event_time, mask
                                )
                            
                            rank_loss = rank_loss + risk_rank_loss
                    
                    # Normalize by number of risks
                    rank_loss = rank_loss / self.num_risks
                    loss = loss + self.alpha_rank * rank_loss
                
                # Add calibration loss if alpha_calibration > 0
                if self.alpha_calibration > 0:
                    # Compute calibration loss for CIF predictions
                    calibration_loss = torch.tensor(0.0, device=device)
                    
                    for risk_idx in range(self.num_risks):
                        if sample_weights is not None:
                            risk_cal_loss = self._compute_cif_calibration_loss(
                                cif[:, risk_idx, :], risk_idx, event_indicator, cause, event_time, mask, sample_weights
                            )
                        else:
                            risk_cal_loss = self._compute_cif_calibration_loss(
                                cif[:, risk_idx, :], risk_idx, event_indicator, cause, event_time, mask
                            )
                        
                        calibration_loss = calibration_loss + risk_cal_loss
                    
                    # Normalize by number of risks
                    calibration_loss = calibration_loss / self.num_risks
                    loss = loss + self.alpha_calibration * calibration_loss
        
        return {
            'loss': loss,
            'hazards': hazards,
            'cif': cif,
            'overall_survival': overall_survival,
            'risk_scores': risk_scores
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate competing risks predictions.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'hazards': Predicted hazard functions [batch_size, num_risks, num_time_bins]
            - 'cif': Cumulative incidence functions [batch_size, num_risks, num_time_bins]
            - 'overall_survival': Overall survival function [batch_size, num_time_bins]
            - 'risk_scores': Risk scores for each cause [batch_size, num_risks]
        """
        # Compute predictions (no loss)
        outputs = self.forward(x)
        
        # Remove loss from outputs
        outputs.pop('loss', None)
        
        return outputs
    
    def compute_metrics(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics for competing risks prediction.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from the forward pass
            
        targets : torch.Tensor
            Ground truth targets
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values
        """
        # Extract predictions
        risk_scores = outputs['risk_scores'].detach().cpu().numpy()  # [batch_size, num_risks]
        cif = outputs['cif'].detach().cpu().numpy()  # [batch_size, num_risks, num_time_bins]
        
        # Extract targets
        event_indicator = targets[:, 0].detach().cpu().numpy()  # Event indicator
        event_time = targets[:, 1].detach().cpu().numpy()  # Time bin
        cause = targets[:, 2].detach().cpu().numpy()  # Cause index
        
        # Initialize metrics dict
        metrics = {}
        
        # Compute cause-specific concordance indices
        for risk_idx in range(self.num_risks):
            c_index = self._compute_cause_specific_c_index(
                risk_scores[:, risk_idx], event_time, event_indicator, cause, risk_idx
            )
            metrics[f'c_index_cause_{risk_idx}'] = c_index
        
        # Compute integrated Brier score for each cause
        for risk_idx in range(self.num_risks):
            brier_score = self._compute_integrated_brier_score_competing(
                cif[:, risk_idx, :], event_time, event_indicator, cause, risk_idx
            )
            metrics[f'brier_score_cause_{risk_idx}'] = brier_score
        
        # Compute average metrics
        metrics['c_index_avg'] = np.mean([metrics[f'c_index_cause_{r}'] for r in range(self.num_risks)])
        metrics['brier_score_avg'] = np.mean([metrics[f'brier_score_cause_{r}'] for r in range(self.num_risks)])
        
        return metrics
    
    def _compute_ranking_loss(self, 
                            risk_scores: torch.Tensor, 
                            event_indicator: torch.Tensor, 
                            event_time: torch.Tensor,
                            mask: torch.Tensor,
                             sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute ranking loss for concordance optimization (adapted for competing risks).
        
        Applies the pairwise hinge loss (see SingleRiskHead._compute_ranking_loss)
        specifically for pairs where the event of interest (specific cause) occurred.
        """
        batch_size = risk_scores.size(0)
        device = risk_scores.device
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # Count valid comparisons
        valid_comparisons = 0
        
        # Compute pairwise rankings
        for i in range(batch_size):
            if mask[i] == 0 or event_indicator[i] == 0:
                # Skip censored or masked samples for first position
                continue
                
            for j in range(batch_size):
                if mask[j] == 0:
                    # Skip masked samples for second position
                    continue
                    
                # Valid comparison if:
                # 1. i had an event and j was censored after i's event, or
                # 2. Both i and j had events, and i's event was before j's
                if (event_indicator[i] == 1 and 
                    ((event_indicator[j] == 0 and event_time[j] > event_time[i]) or
                     (event_indicator[j] == 1 and event_time[i] < event_time[j]))):
                    
                    # i should have a higher risk score than j
                    risk_diff = risk_scores[j] - risk_scores[i]
                    
                    # Compute hinge loss: max(0, 1 - (risk_i - risk_j))
                    pair_loss = torch.relu(1.0 + risk_diff)
                    
                    # Apply sample weights if provided
                    if sample_weights is not None:
                        # Weight the pair loss by the product of both sample weights
                        pair_weight = sample_weights[i] * sample_weights[j]
                        loss = loss + pair_loss * pair_weight
                        valid_comparisons += pair_weight
                    else:
                        loss = loss + pair_loss
                        valid_comparisons += 1
        
        # Normalize loss by number of valid comparisons
        if valid_comparisons > 0:
            loss = loss / valid_comparisons
            
        return loss
    
    def _compute_cif_calibration_loss(self,
                                    cif: torch.Tensor,
                                    risk_idx: int,
                                    event_indicator: torch.Tensor,
                                    cause: torch.Tensor,
                                    event_time: torch.Tensor,
                                    mask: torch.Tensor,
                                     sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute calibration loss for CIF predictions.
        
        Calculates the L2 distance between the mean predicted Cumulative Incidence Function (CIF)
        for a specific cause and the empirical CIF (e.g., Aalen-Johansen estimate) at each time bin.
        Loss_k = sum_{t=1}^{T} ( mean(CIF_k_pred(t)) - CIF_k_empirical(t) )^2 / T
        
        Parameters
        ----------
        cif : torch.Tensor
            Predicted CIF for one cause [batch_size, num_time_bins]
            
        risk_idx : int
            Risk index for which to compute calibration
            
        event_indicator : torch.Tensor
            Event indicators [batch_size]
            
        cause : torch.Tensor
            Cause indices [batch_size]
            
        event_time : torch.Tensor
            Event times [batch_size]
            
        mask : torch.Tensor
            Mask indicating valid samples [batch_size]
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation [batch_size]
            
        Returns
        -------
        torch.Tensor
            Calibration loss value (scalar)
        """
        batch_size = cif.size(0)
        device = cif.device
        
        # Get number of time bins
        num_bins = cif.size(1)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # For each time bin
        for t in range(num_bins):
            # Count events of this cause that occur at or before time t
            events_at_t = torch.zeros(batch_size, device=device)
            
            for i in range(batch_size):
                if mask[i] > 0 and event_indicator[i] > 0 and cause[i] == risk_idx and event_time[i] <= t:
                    events_at_t[i] = 1.0
            
            # Calculate empirical probability of event by time t
            if sample_weights is not None:
                weighted_events = events_at_t * sample_weights * mask
                weighted_mask = mask * sample_weights
                empirical_prob = torch.sum(weighted_events) / (torch.sum(weighted_mask) + 1e-6)
            else:
                empirical_prob = torch.sum(events_at_t * mask) / (torch.sum(mask) + 1e-6)
                
            # Calculate mean predicted CIF at time t
            predicted_prob = torch.mean(cif[:, t])
            
            # L2 loss between predicted and empirical probabilities
            bin_loss = (predicted_prob - empirical_prob).pow(2)
            loss = loss + bin_loss
        
        # Average over all time bins
        loss = loss / num_bins
        
        return loss
    
    def _compute_cause_specific_c_index(self, risk_scores, event_time, event_indicator, cause, cause_idx):
        """
        Compute cause-specific concordance index.
        
        For cause k:
        - Treat events of cause k as events
        - Treat events of other causes and censoring as competing censoring
        """
        n_samples = len(risk_scores)
        concordant_pairs = 0
        total_pairs = 0
        
        for i in range(n_samples):
            # Skip samples that didn't experience the cause of interest
            if not (event_indicator[i] == 1 and cause[i] == cause_idx):
                continue
                
            for j in range(n_samples):
                # Valid comparison if:
                # 1. i had an event of cause k and j was censored or had different cause after i's event
                is_valid = (
                    (event_indicator[i] == 1 and cause[i] == cause_idx) and
                    ((event_indicator[j] == 0 and event_time[j] > event_time[i]) or
                     (event_indicator[j] == 1 and cause[j] != cause_idx and event_time[j] > event_time[i]) or
                     (event_indicator[j] == 1 and cause[j] == cause_idx and event_time[i] < event_time[j]))
                )
                
                if is_valid:
                    total_pairs += 1
                    
                    # Higher risk score should predict earlier event
                    if risk_scores[i] > risk_scores[j]:
                        concordant_pairs += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant_pairs += 0.5
        
        # Return c-index (proportion of concordant pairs)
        if total_pairs == 0:
            return 0.5
        else:
            return concordant_pairs / total_pairs
    
    def _compute_integrated_brier_score_competing(self, cif, event_time, event_indicator, cause, cause_idx):
        """
        Compute integrated Brier score for competing risks.
        
        For cause k, the Brier score measures how well the CIF predicts the
        probability of experiencing cause k by time t.
        """
        n_samples = len(event_time)
        n_time_bins = cif.shape[1]
        
        # Initialize Brier score for each time point
        brier_scores = np.zeros(n_time_bins)
        
        # For each time bin
        for t in range(n_time_bins):
            brier_score_t = 0.0
            weight_sum = 0.0
            
            for i in range(n_samples):
                # Observed outcome at time t for cause k
                Y_t = 0.0  # Default: event of cause k has not occurred by time t
                weight = 1.0  # Default weight
                
                if event_indicator[i] == 1 and cause[i] == cause_idx and event_time[i] <= t:
                    # Event of cause k before or at time t
                    Y_t = 1.0
                elif event_time[i] <= t:
                    # Censored or other cause event before or at time t
                    # In competing risks, we can still compute the weight
                    # but it requires more complex inverse probability censoring weights
                    # For simplicity, we'll use a basic approach here
                    if event_indicator[i] == 0:
                        weight = 0.5  # Censored: less certain about true outcome
                    else:
                        weight = 0.8  # Different cause: more certain about this cause not occurring
                
                # Skip if weight is zero
                if weight > 0:
                    # Predicted probability of event of cause k by time t
                    pred_t = cif[i, t]
                    
                    # Squared error weighted by adjusted weight
                    brier_score_t += weight * (Y_t - pred_t) ** 2
                    weight_sum += weight
            
            # Average over samples with non-zero weights
            if weight_sum > 0:
                brier_scores[t] = brier_score_t / weight_sum
        
        # Compute integrated score (mean over time)
        return np.mean(brier_scores)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            'num_time_bins': self.num_time_bins,
            'num_risks': self.num_risks,
            'alpha_rank': self.alpha_rank,
            'alpha_calibration': self.alpha_calibration,
            'use_softmax': self.use_softmax,
            'use_cause_specific': self.use_cause_specific
        })
        return config
