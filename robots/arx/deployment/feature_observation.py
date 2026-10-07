"""Expected missing task observations, distinct from device/implementation faults."""


class FeatureObservationUnavailable(ValueError):
    def __init__(self, reason, *, available, unavailable, reason_code,
                 last_valid_target_monotonic_ns=None):
        super().__init__(reason)
        self.available = dict(available)
        self.unavailable = tuple(unavailable)
        self.detail = {
            "status": "unknown", "reason_code": reason_code, "reason": reason,
            "unavailable_features": list(unavailable),
            "last_valid_target_monotonic_ns": last_valid_target_monotonic_ns,
        }
