"""Typed exceptions and the shared set of error codes used on the wire.

Error codes travel from the daemon to the CLI/waiter inside ``error`` messages
(see :mod:`gpuq.protocol`). Keeping them in one place lets both sides agree on
the vocabulary and lets the CLI present a clean message in the user's shell.
"""


class GpuqError(Exception):
    """Base class for all gpuq errors."""


class ConfigError(GpuqError):
    """Raised when the configuration file or overrides are invalid."""


class ProtocolError(GpuqError):
    """Raised when a message on the wire is malformed or unexpected."""


class ValidationError(GpuqError):
    """Raised when a submission is rejected.

    Carries a machine-readable ``code`` (one of the constants below) so the
    daemon can relay it to the client and the client can exit non-zero with a
    helpful message.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# --- Error codes (shared CLI <-> daemon vocabulary) -----------------------

# Submission was rejected because --share was given without --mem.
ERR_SHARE_REQUIRES_MEM = "share_requires_mem"
# --mem could not be parsed (e.g. missing unit suffix).
ERR_BAD_MEM = "bad_mem"
# --gpus was not a positive integer.
ERR_BAD_GPUS = "bad_gpus"
# The job requested more GPUs than the daemon manages.
ERR_TOO_MANY_GPUS = "too_many_gpus"
# --mem exceeds the largest managed GPU; the job could never be placed.
ERR_MEM_TOO_LARGE = "mem_too_large"
# A cancel/cancel_all was refused (not the owner, or job already running).
ERR_PERMISSION = "permission_denied"
# A referenced job id does not exist.
ERR_NOT_FOUND = "not_found"
# A job could not be cancelled because it already finished.
ERR_NOT_CANCELLABLE = "not_cancellable"
# A running job was cancelled without force; the client must confirm first.
ERR_CANCEL_REQUIRES_FORCE = "cancel_requires_force"
# Catch-all for malformed requests.
ERR_BAD_REQUEST = "bad_request"
