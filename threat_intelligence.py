import numpy as np

from config import Config

class ThreatIntelligence:
    def __init__(self):
        self.anomaly_history = []
        self.ewma_score = 0.0

    def calculate_early_warning_score(self, reconstruction_errors):
        """
        Takes mean reconstruction error for a batch, updates exponential moving average,
        and returns the proactive Early Warning Score.
        """
        current_error = float(np.mean(reconstruction_errors))
        self.anomaly_history.append(current_error)

        # Calculate EWMA: S_t = alpha * e_t + (1 - alpha) * S_{t-1}
        alpha = Config.EARLY_WARNING_ALPHA
        self.ewma_score = alpha * current_error + (1 - alpha) * self.ewma_score

        is_critical = self.ewma_score > Config.EARLY_WARNING_CRITICAL_SCORE
        return self.ewma_score, is_critical

    @staticmethod
    def map_to_mitre_and_stride(anomalous_features, stage_id):
        """
        Maps top anomalous features to MITRE ATT&CK for ICS and STRIDE.
        In our synthetic SWaT data:
        F0: Main level/Flow (Impact/Manipulation)
        F1: Tank Pressure (Tampering/Impact)
        F2: Pump State (Execution/Manipulation of Control)
        F3: Chem Dosage (Impact)
        """
        alerts = []
        for feature in anomalous_features:
            # Determine mapping by prefix
            prefix = ''.join([c for c in feature if c.isalpha()])
            
            if prefix == 'FIT' or prefix == 'DPIT':
                meta = {'STRIDE': 'Spoofing/Tampering', 'MITRE': 'T0832 (Manipulation of View)'}
            elif prefix == 'LIT':
                meta = {'STRIDE': 'Spoofing', 'MITRE': 'TA0105 (Impact - Loss of View)'}
            elif prefix == 'MV':
                meta = {'STRIDE': 'Elevation of Privilege', 'MITRE': 'T0831 (Manipulation of Control)'}
            elif prefix == 'P':
                meta = {'STRIDE': 'Denial of Service', 'MITRE': 'TA0104 (Execution)'}
            elif prefix == 'AIT':
                meta = {'STRIDE': 'Tampering', 'MITRE': 'T0831 (Manipulation of Control)'}
            else:
                meta = {'STRIDE': 'Information Disclosure', 'MITRE': 'T0856 (Spoofing Reporting)'}

            alerts.append({
                'Stage': f'P{stage_id + 1}',
                'Affected Component': feature,
                'STRIDE Threat': meta['STRIDE'],
                'MITRE Class': meta['MITRE']
            })
            
        return alerts
