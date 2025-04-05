import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from enhanced_deephit.models.tasks.base import TaskHead


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
        
    def forward(self, 
               x: torch.Tensor, 
               targets: Optional[torch.Tensor] = None, 
               mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
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
        
        # Compute survival function (cumulative product of 1 - hazard)
        # S(t) = \prod_{j=1}^{t} (1 - h(j))
        survival = torch.cumprod(1 - hazards, dim=1)
        
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
                    # Negative log-likelihood loss
                    # For observed events:
                    #   - log(hazard(event_time)) + sum_{j=1}^{event_time-1} log(1 - hazard(j))
                    # For censored:
                    #   - sum_{j=1}^{censor_time} log(1 - hazard(j))
                    
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
                    
                    # Compute mean NLL for valid samples
                    nll = torch.sum(nll * mask) / (valid_samples + 1e-6)
                    loss = nll
                
                # Add ranking loss if alpha_rank > 0
                if self.alpha_rank > 0:
                    # Compute concordance loss
                    rank_loss = self._compute_ranking_loss(risk_score, event_indicator, event_time, mask)
                    loss = loss + self.alpha_rank * rank_loss
                
                # Add calibration loss if alpha_calibration > 0
                if self.alpha_calibration > 0:
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
            Ground truth targets with shape [batch_size, 2 + num_time_bins]:
            - targets[:, 0]: Event indicator (1 if event occurred, 0 if censored)
            - targets[:, 1]: Time bin index where event/censoring occurred
            - targets[:, 2:]: One-hot encoding of event time (for convenience)
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values:
            - 'c_index': Concordance index
            - 'brier_score': Integrated Brier score
            - 'auc': Time-dependent AUC
        """
        # Extract predictions and targets
        risk_scores = outputs['risk_score'].detach().cpu().numpy()
        survival = outputs['survival'].detach().cpu().numpy()
        
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
                             mask: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss for concordance optimization.
        
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
                                 mask: torch.Tensor) -> torch.Tensor:
        """
        Compute calibration loss to ensure predicted probabilities match empirical frequencies.
        
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
        
        # For each time bin, compute expected vs observed event counts
        for t in range(num_bins):
            # Samples at risk at time t (not having events before t)
            at_risk = (event_time >= t) & (mask > 0)
            num_at_risk = torch.sum(at_risk)
            
            if num_at_risk > 0:
                # Expected number of events at time t based on predicted hazards
                hazard_t = 1.0 - survival[:, t] / torch.where(t > 0, survival[:, t-1], torch.ones_like(survival[:, 0]))
                expected_events = torch.sum(hazard_t[at_risk])
                
                # Observed number of events at time t
                observed_events = torch.sum((event_time == t) & (event_indicator == 1) & at_risk)
                
                # Squared difference between expected and observed
                bin_loss = (expected_events - observed_events).pow(2) / num_at_risk
                loss = loss + bin_loss
        
        # Normalize by number of time bins
        loss = loss / num_bins
        
        return loss
    
    def _compute_c_index(self, risk_scores: np.ndarray, event_time: np.ndarray, event_indicator: np.ndarray) -> float:
        """
        Compute concordance index for survival predictions.
        
        Parameters
        ----------
        risk_scores : np.ndarray
            Predicted risk scores
            
        event_time : np.ndarray
            Event times
            
        event_indicator : np.ndarray
            Event indicators (1 if event occurred, 0 if censored)
            
        Returns
        -------
        float
            Concordance index
        """
        # For convenience, using NumPy for this computation
        n_samples = len(risk_scores)
        
        # Initialize counters
        concordant = 0
        discordant = 0
        tied_risk = 0
        
        # Count comparable pairs
        comparable_pairs = 0
        
        # Compute pairwise comparisons
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
                    
                    # i should have a higher risk score than j
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied_risk += 1
                    
                    comparable_pairs += 1
        
        # Compute concordance index
        if comparable_pairs > 0:
            return (concordant + 0.5 * tied_risk) / comparable_pairs
        else:
            return 0.5  # Default value when no comparable pairs exist
    
    def _compute_integrated_brier_score(self, survival: np.ndarray, event_time: np.ndarray, event_indicator: np.ndarray) -> float:
        """
        Compute integrated Brier score for survival predictions.
        
        Parameters
        ----------
        survival : np.ndarray
            Predicted survival functions [batch_size, num_time_bins]
            
        event_time : np.ndarray
            Event times
            
        event_indicator : np.ndarray
            Event indicators (1 if event occurred, 0 if censored)
            
        Returns
        -------
        float
            Integrated Brier score
        """
        # Number of time bins
        num_bins = survival.shape[1]
        
        # Initialize Brier scores for each time point
        brier_scores = np.zeros(num_bins)
        
        # Compute Brier score at each time point
        for t in range(num_bins):
            # Observed survival status at time t
            # 1 if alive at time t, 0 if event before t
            observed = (event_time > t).astype(float)
            
            # For censored before t, we don't have ground truth
            # Exclude these from the computation
            mask = (event_indicator == 1) | (event_time > t)
            
            if np.sum(mask) > 0:
                # Compute squared error between predicted and observed
                brier_scores[t] = np.mean(((survival[:, t] - observed) ** 2)[mask])
        
        # Compute mean Brier score across all time points
        return np.mean(brier_scores)
    
    def _compute_time_dependent_auc(self, risk_scores: np.ndarray, event_time: np.ndarray, event_indicator: np.ndarray) -> float:
        """
        Compute time-dependent AUC for survival predictions.
        
        Parameters
        ----------
        risk_scores : np.ndarray
            Predicted risk scores
            
        event_time : np.ndarray
            Event times
            
        event_indicator : np.ndarray
            Event indicators (1 if event occurred, 0 if censored)
            
        Returns
        -------
        float
            Mean time-dependent AUC
        """
        # Number of unique event times
        unique_times = np.unique(event_time[event_indicator == 1])
        
        if len(unique_times) == 0:
            return 0.5  # Default value when no events
        
        # Initialize AUCs for each time point
        aucs = np.zeros(len(unique_times))
        
        # Compute AUC at each time point
        for i, t in enumerate(unique_times):
            # Positive class: samples with events at time t
            positives = (event_time == t) & (event_indicator == 1)
            
            # Negative class: samples alive after time t
            negatives = event_time > t
            
            # Skip if no positives or negatives
            if np.sum(positives) == 0 or np.sum(negatives) == 0:
                aucs[i] = 0.5
                continue
            
            # Extract risk scores for positives and negatives
            pos_scores = risk_scores[positives]
            neg_scores = risk_scores[negatives]
            
            # Compute AUC by comparing all pairs
            n_pos = len(pos_scores)
            n_neg = len(neg_scores)
            
            # Count concordant pairs
            concordant = 0
            
            for pos_score in pos_scores:
                concordant += np.sum(pos_score > neg_scores)
                concordant += 0.5 * np.sum(pos_score == neg_scores)
            
            # Compute AUC
            aucs[i] = concordant / (n_pos * n_neg)
        
        # Return mean AUC across all time points
        return np.mean(aucs)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
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
    
    Predicts the probability of different event types occurring at each time point,
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
                use_cause_specific: bool = False,
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
            Number of competing risks (event types)
            
        alpha_rank : float, default=0.0
            Weight of the ranking loss component
            
        alpha_calibration : float, default=0.0
            Weight of the calibration loss component
            
        task_weight : float, default=1.0
            Weight of this task in the multi-task loss
            
        use_softmax : bool, default=True
            Whether to use softmax for cause probabilities (True) or independent sigmoids (False)
            
        use_cause_specific : bool, default=False
            Whether to use cause-specific networks (True) or shared network (False)
            
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
        if use_cause_specific:
            # Separate network for each cause
            self.shared_network = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.LayerNorm(input_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
            self.cause_networks = nn.ModuleList([
                nn.Linear(input_dim * 2, num_time_bins) 
                for _ in range(num_risks)
            ])
        else:
            # Single network with multi-output head
            self.prediction_network = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.LayerNorm(input_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim * 2, num_time_bins * num_risks)
            )
            
    def forward(self, 
               x: torch.Tensor, 
               targets: Optional[torch.Tensor] = None, 
               mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for competing risks prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        targets : torch.Tensor, optional
            Target values with shape [batch_size, 2 + num_risks * num_time_bins]:
            - targets[:, 0]: Event indicator (1 if any event occurred, 0 if censored)
            - targets[:, 1]: Time bin index where event/censoring occurred
            - targets[:, 2]: Cause index (0 to num_risks-1, or -1 if censored)
            - targets[:, 3:]: One-hot encoding of event time and cause (for convenience)
            
        mask : torch.Tensor, optional
            Mask indicating which samples have targets for this task
            [batch_size], where 1 means target is available
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'loss': Loss for this task (if targets provided)
            - 'cause_hazards': Predicted cause-specific hazards [batch_size, num_risks, num_time_bins]
            - 'overall_survival': Predicted overall survival function [batch_size, num_time_bins]
            - 'cif': Cumulative incidence functions [batch_size, num_risks, num_time_bins]
            - 'risk_scores': Cause-specific risk scores [batch_size, num_risks]
        """
        # Get batch size and device
        batch_size = x.size(0)
        device = x.device
        
        # Compute raw predictions
        if self.use_cause_specific:
            # Shared representation
            shared_features = self.shared_network(x)
            
            # Cause-specific predictions
            cause_logits = []
            for i in range(self.num_risks):
                cause_logits.append(self.cause_networks[i](shared_features))
                
            # Stack along cause dimension
            cause_logits = torch.stack(cause_logits, dim=1)  # [batch_size, num_risks, num_time_bins]
        else:
            # Single network prediction
            outputs = self.prediction_network(x)
            
            # Reshape to [batch_size, num_risks, num_time_bins]
            cause_logits = outputs.view(batch_size, self.num_risks, self.num_time_bins)
        
        # Compute cause-specific hazards
        if self.use_softmax:
            # Add an additional dimension for "no event"
            # [batch_size, num_risks + 1, num_time_bins]
            padded_logits = torch.cat([
                torch.zeros(batch_size, 1, self.num_time_bins, device=device),
                cause_logits
            ], dim=1)
            
            # Apply softmax for each time point
            # This enforces that at each time point, probabilities sum to 1 across all causes (including no event)
            probs = F.softmax(padded_logits, dim=1)
            
            # Extract cause-specific hazards (excluding "no event")
            cause_hazards = probs[:, 1:, :]
            
            # Probability of no event
            no_event_prob = probs[:, 0, :]
        else:
            # Apply sigmoid for independent cause-specific hazards
            cause_hazards = torch.sigmoid(cause_logits)
            
            # Probability of no event = probability of not having any of the events
            no_event_prob = torch.prod(1 - cause_hazards, dim=1)
        
        # Compute overall survival function
        # S(t) = \prod_{j=1}^{t} (probability of no event at time j)
        overall_survival = torch.cumprod(no_event_prob, dim=1)
        
        # Compute cumulative incidence functions for each cause
        # CIF_k(t) = \sum_{j=1}^{t} S(j-1) * cause_hazards_k(j)
        # Initialize CIF starting at 0 for all risks
        cif = torch.zeros(batch_size, self.num_risks, self.num_time_bins, device=device)
        
        # Calculate CIF for each time point
        for t in range(self.num_time_bins):
            # Survival up to previous time point
            prev_survival = torch.ones(batch_size, device=device) if t == 0 else overall_survival[:, t-1]
            
            # Probability of event of each type at time t
            event_probs = cause_hazards[:, :, t] * prev_survival.unsqueeze(1)
            
            # Add to cumulative incidence (start from 0)
            if t == 0:
                cif[:, :, t] = event_probs
            else:
                cif[:, :, t] = cif[:, :, t-1] + event_probs
        
        # Compute cause-specific risk scores (negative expected survival time for each cause)
        risk_scores = torch.sum(cif, dim=2)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # If targets are provided, compute the loss
        if targets is not None:
            # Extract event indicator, time index, and cause
            event_indicator = targets[:, 0]
            event_time = targets[:, 1].long()
            event_cause = targets[:, 2].long()  # -1 if censored
            
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
            
            # Skip loss computation if no valid samples
            if valid_samples > 0:
                # Negative log-likelihood loss
                nll = torch.zeros(batch_size, device=device)
                
                # Compute likelihood for each sample
                for i in range(batch_size):
                    if mask[i] > 0:  # If this sample has a valid target
                        t = event_time[i]
                        
                        if event_indicator[i] > 0:  # Event occurred
                            # Add log probability of the specific cause at time t
                            cause = event_cause[i]
                            
                            # Probability of this specific cause at time t
                            if t > 0:
                                cause_prob = cause_hazards[i, cause, t] * overall_survival[i, t-1]
                            else:
                                cause_prob = cause_hazards[i, cause, t]
                                
                            nll[i] = -torch.log(cause_prob + 1e-7)
                        else:  # Censored
                            # Add log probability of survival up to censoring time
                            nll[i] = -torch.log(overall_survival[i, t] + 1e-7)
                
                # Compute mean NLL for valid samples
                nll = torch.sum(nll * mask) / (valid_samples + 1e-6)
                loss = nll
                
                # Add ranking loss if alpha_rank > 0
                if self.alpha_rank > 0:
                    # Compute concordance loss for each cause
                    rank_loss = torch.tensor(0.0, device=device)
                    
                    for cause in range(self.num_risks):
                        # Extract samples with this cause
                        cause_mask = (event_cause == cause) & (mask > 0)
                        
                        if torch.sum(cause_mask) > 0:
                            cause_rank_loss = self._compute_ranking_loss(
                                risk_scores[:, cause], 
                                cause_mask.float(), 
                                event_time,
                                mask
                            )
                            rank_loss = rank_loss + cause_rank_loss
                    
                    # Average across causes
                    rank_loss = rank_loss / self.num_risks
                    loss = loss + self.alpha_rank * rank_loss
                
                # Add calibration loss if alpha_calibration > 0
                if self.alpha_calibration > 0:
                    calibration_loss = self._compute_calibration_loss(
                        cif, 
                        event_indicator, 
                        event_time, 
                        event_cause,
                        mask
                    )
                    loss = loss + self.alpha_calibration * calibration_loss
        
        return {
            'loss': loss,
            'cause_hazards': cause_hazards,
            'overall_survival': overall_survival,
            'cif': cif,
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
            - 'cause_hazards': Predicted cause-specific hazards [batch_size, num_risks, num_time_bins]
            - 'overall_survival': Predicted overall survival function [batch_size, num_time_bins]
            - 'cif': Cumulative incidence functions [batch_size, num_risks, num_time_bins]
            - 'risk_scores': Cause-specific risk scores [batch_size, num_risks]
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
            Ground truth targets with shape [batch_size, 3 + num_risks * num_time_bins]:
            - targets[:, 0]: Event indicator (1 if any event occurred, 0 if censored)
            - targets[:, 1]: Time bin index where event/censoring occurred
            - targets[:, 2]: Cause index (0 to num_risks-1, or -1 if censored)
            - targets[:, 3:]: One-hot encoding of event time and cause (for convenience)
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values:
            - 'cause_specific_c_index_{i}': Cause-specific concordance index for cause i
            - 'cause_specific_auc_{i}': Time-dependent AUC for cause i
            - 'overall_c_index': Overall concordance index
            - 'integrated_brier_score': Integrated Brier score
        """
        # Extract predictions and targets
        risk_scores = outputs['risk_scores'].detach().cpu().numpy()
        cif = outputs['cif'].detach().cpu().numpy()
        
        event_indicator = targets[:, 0].detach().cpu().numpy()
        event_time = targets[:, 1].detach().cpu().numpy()
        event_cause = targets[:, 2].detach().cpu().numpy()
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Compute cause-specific metrics
        for i in range(self.num_risks):
            # Compute cause-specific concordance index
            cause_c_index = self._compute_cause_specific_c_index(
                risk_scores[:, i], 
                event_time, 
                event_indicator, 
                event_cause, 
                i
            )
            metrics[f'cause_specific_c_index_{i}'] = cause_c_index
            
            # Compute cause-specific AUC
            cause_auc = self._compute_cause_specific_auc(
                risk_scores[:, i], 
                event_time, 
                event_indicator, 
                event_cause, 
                i
            )
            metrics[f'cause_specific_auc_{i}'] = cause_auc
        
        # Compute overall concordance index
        metrics['overall_c_index'] = np.mean([metrics[f'cause_specific_c_index_{i}'] for i in range(self.num_risks)])
        
        # Compute integrated Brier score
        metrics['integrated_brier_score'] = self._compute_integrated_brier_score(
            cif, 
            event_time, 
            event_indicator, 
            event_cause
        )
        
        return metrics
    
    def _compute_ranking_loss(self, 
                             risk_scores: torch.Tensor, 
                             cause_mask: torch.Tensor, 
                             event_time: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss for cause-specific concordance optimization.
        
        Parameters
        ----------
        risk_scores : torch.Tensor
            Predicted risk scores for a specific cause [batch_size]
            
        cause_mask : torch.Tensor
            Mask indicating samples with events of this cause [batch_size]
            
        event_time : torch.Tensor
            Event times [batch_size]
            
        mask : torch.Tensor
            Mask indicating valid samples [batch_size]
            
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
            if cause_mask[i] == 0:
                # Skip samples without this cause for first position
                continue
                
            for j in range(batch_size):
                if mask[j] == 0:
                    # Skip masked samples for second position
                    continue
                    
                # Valid comparison if:
                # 1. i had an event of this cause and j was censored after i's event, or
                # 2. i had an event of this cause and j had an event of any cause after i's event
                if (cause_mask[i] == 1 and event_time[j] > event_time[i]):
                    # i should have a higher risk score than j
                    risk_diff = risk_scores[j] - risk_scores[i]
                    
                    # Compute hinge loss: max(0, 1 - (risk_i - risk_j))
                    pair_loss = torch.relu(1.0 + risk_diff)
                    
                    loss = loss + pair_loss
                    valid_comparisons += 1
        
        # Normalize loss by number of valid comparisons
        if valid_comparisons > 0:
            loss = loss / valid_comparisons
            
        return loss
    
    def _compute_calibration_loss(self,
                                 cif: torch.Tensor,
                                 event_indicator: torch.Tensor,
                                 event_time: torch.Tensor,
                                 event_cause: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
        """
        Compute calibration loss to ensure predicted CIFs match empirical frequencies.
        
        Parameters
        ----------
        cif : torch.Tensor
            Predicted cumulative incidence functions [batch_size, num_risks, num_time_bins]
            
        event_indicator : torch.Tensor
            Event indicators [batch_size]
            
        event_time : torch.Tensor
            Event times [batch_size]
            
        event_cause : torch.Tensor
            Event causes [batch_size]
            
        mask : torch.Tensor
            Mask indicating valid samples [batch_size]
            
        Returns
        -------
        torch.Tensor
            Calibration loss value (scalar)
        """
        batch_size = cif.size(0)
        device = cif.device
        
        # Get number of time bins and risks
        num_risks = cif.size(1)
        num_bins = cif.size(2)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # For each cause and time bin, compute expected vs observed event counts
        for cause in range(num_risks):
            for t in range(num_bins):
                # Samples at risk at time t (not having events before t)
                at_risk = (event_time >= t) & (mask > 0)
                num_at_risk = torch.sum(at_risk)
                
                if num_at_risk > 0:
                    # Expected number of events of this cause at time t based on predicted CIFs
                    if t > 0:
                        expected_events = torch.sum((cif[:, cause, t] - cif[:, cause, t-1])[at_risk])
                    else:
                        expected_events = torch.sum(cif[:, cause, t][at_risk])
                    
                    # Observed number of events of this cause at time t
                    observed_events = torch.sum((event_time == t) & (event_cause == cause) & at_risk)
                    
                    # Squared difference between expected and observed
                    bin_loss = (expected_events - observed_events).pow(2) / num_at_risk
                    loss = loss + bin_loss
        
        # Normalize by number of time bins and causes
        loss = loss / (num_risks * num_bins)
        
        return loss
    
    def _compute_cause_specific_c_index(self, 
                                       risk_scores: np.ndarray, 
                                       event_time: np.ndarray, 
                                       event_indicator: np.ndarray,
                                       event_cause: np.ndarray,
                                       cause: int) -> float:
        """
        Compute cause-specific concordance index for competing risks.
        
        Parameters
        ----------
        risk_scores : np.ndarray
            Predicted risk scores for a specific cause
            
        event_time : np.ndarray
            Event times
            
        event_indicator : np.ndarray
            Event indicators (1 if event occurred, 0 if censored)
            
        event_cause : np.ndarray
            Event causes (-1 if censored)
            
        cause : int
            The specific cause to evaluate
            
        Returns
        -------
        float
            Cause-specific concordance index
        """
        # For convenience, using NumPy for this computation
        n_samples = len(risk_scores)
        
        # Initialize counters
        concordant = 0
        discordant = 0
        tied_risk = 0
        
        # Count comparable pairs
        comparable_pairs = 0
        
        # Compute pairwise comparisons
        for i in range(n_samples):
            # Skip samples without this cause for first position
            if not (event_indicator[i] == 1 and event_cause[i] == cause):
                continue
                
            for j in range(n_samples):
                # Valid comparison if:
                # 1. i had an event of this cause and j was censored after i's event, or
                # 2. i had an event of this cause and j had an event of any cause after i's event
                if event_time[j] > event_time[i]:
                    # i should have a higher risk score than j
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied_risk += 1
                    
                    comparable_pairs += 1
        
        # Compute concordance index
        if comparable_pairs > 0:
            return (concordant + 0.5 * tied_risk) / comparable_pairs
        else:
            return 0.5  # Default value when no comparable pairs exist
    
    def _compute_cause_specific_auc(self, 
                                   risk_scores: np.ndarray, 
                                   event_time: np.ndarray, 
                                   event_indicator: np.ndarray,
                                   event_cause: np.ndarray,
                                   cause: int) -> float:
        """
        Compute cause-specific time-dependent AUC for competing risks.
        
        Parameters
        ----------
        risk_scores : np.ndarray
            Predicted risk scores for a specific cause
            
        event_time : np.ndarray
            Event times
            
        event_indicator : np.ndarray
            Event indicators (1 if event occurred, 0 if censored)
            
        event_cause : np.ndarray
            Event causes (-1 if censored)
            
        cause : int
            The specific cause to evaluate
            
        Returns
        -------
        float
            Cause-specific time-dependent AUC
        """
        # Find unique event times for this cause
        cause_events = (event_indicator == 1) & (event_cause == cause)
        unique_times = np.unique(event_time[cause_events])
        
        if len(unique_times) == 0:
            return 0.5  # Default value when no events of this cause
        
        # Initialize AUCs for each time point
        aucs = np.zeros(len(unique_times))
        
        # Compute AUC at each time point
        for i, t in enumerate(unique_times):
            # Positive class: samples with events of this cause at time t
            positives = (event_time == t) & (event_indicator == 1) & (event_cause == cause)
            
            # Negative class: samples alive after time t
            negatives = event_time > t
            
            # Skip if no positives or negatives
            if np.sum(positives) == 0 or np.sum(negatives) == 0:
                aucs[i] = 0.5
                continue
            
            # Extract risk scores for positives and negatives
            pos_scores = risk_scores[positives]
            neg_scores = risk_scores[negatives]
            
            # Compute AUC by comparing all pairs
            n_pos = len(pos_scores)
            n_neg = len(neg_scores)
            
            # Count concordant pairs
            concordant = 0
            
            for pos_score in pos_scores:
                concordant += np.sum(pos_score > neg_scores)
                concordant += 0.5 * np.sum(pos_score == neg_scores)
            
            # Compute AUC
            aucs[i] = concordant / (n_pos * n_neg)
        
        # Return mean AUC across all time points
        return np.mean(aucs)
    
    def _compute_integrated_brier_score(self, 
                                       cif: np.ndarray, 
                                       event_time: np.ndarray, 
                                       event_indicator: np.ndarray,
                                       event_cause: np.ndarray) -> float:
        """
        Compute integrated Brier score for competing risks predictions.
        
        Parameters
        ----------
        cif : np.ndarray
            Predicted cumulative incidence functions [batch_size, num_risks, num_time_bins]
            
        event_time : np.ndarray
            Event times
            
        event_indicator : np.ndarray
            Event indicators (1 if event occurred, 0 if censored)
            
        event_cause : np.ndarray
            Event causes (-1 if censored)
            
        Returns
        -------
        float
            Integrated Brier score
        """
        # Number of time bins and risks
        num_risks = cif.shape[1]
        num_bins = cif.shape[2]
        
        # Initialize Brier scores
        brier_scores = np.zeros((num_risks, num_bins))
        
        # Compute Brier score for each cause and time point
        for cause in range(num_risks):
            for t in range(num_bins):
                # True status for this cause at time t
                # 1 if event of this cause by time t, 0 otherwise
                true_status = (event_indicator == 1) & (event_cause == cause) & (event_time <= t)
                
                # Mask for samples that we can evaluate
                # (either had an event by time t or were still being followed at time t)
                mask = (event_indicator == 1) & (event_time <= t) | (event_time > t)
                
                if np.sum(mask) > 0:
                    # Compute squared error between predicted and true
                    pred_cif = cif[:, cause, t]
                    squared_error = ((pred_cif - true_status.astype(float)) ** 2)[mask]
                    brier_scores[cause, t] = np.mean(squared_error)
        
        # Compute mean Brier score across all time points and causes
        return np.mean(brier_scores)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
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