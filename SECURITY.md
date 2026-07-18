# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.
Instead, e-mail the maintainers at **security@fujicv.example.com** with:

1. A description of the vulnerability and its potential impact.
2. Steps to reproduce or a minimal proof-of-concept.
3. Any suggested mitigations you have identified.

We aim to acknowledge reports within **48 hours** and to publish a patch
within **14 days** for confirmed critical issues.

---

## Security Design Principles

### API Keys and Credentials

- **No hardcoded secrets.** FujiCV never embeds API keys, passwords, or
  tokens in source code, configuration files, example scripts, or
  documentation.
- **W&B authentication via environment variable only.** The `WandbLogger`
  class reads the Weights & Biases API key exclusively from the
  `WANDB_API_KEY` environment variable.  The key is never accepted as a
  constructor argument, stored as an instance attribute, logged to files, or
  transmitted through any channel other than the official `wandb` library.
- **Config files must not contain secrets.** YAML configuration files
  (including the examples shipped with this package) must not include
  API keys, database credentials, or any other sensitive values.

### Loss and Metric Functions

- **Pure tensor operations, no network calls.** All loss functions
  (`fujicv.losses`) and metric functions (`fujicv.metrics`) operate
  exclusively on in-memory tensors or numpy arrays.  They never make
  outbound network requests, read from or write to disk, or spawn
  sub-processes.

### Dependencies

- FujiCV pins minimum versions for all dependencies.  Users are encouraged
  to keep dependencies up-to-date and to run `pip audit` or equivalent tools
  to detect known CVEs in the dependency tree.
- The CI pipeline runs `detect-secrets` on every pull request to prevent
  accidental secret commits.

### Data Handling

- FujiCV does not collect, transmit, or store user data.  All processing
  occurs locally on the user's machine.
- When W&B logging is enabled, data sent to Weights & Biases is governed
  by the W&B privacy policy.  Users are responsible for ensuring they do
  not log personally identifiable information (PII) in metric dictionaries.

### ONNX Export

- The `export.to_onnx` and `export.verify_onnx` functions operate on
  local files only.  They do not upload models to any external service.
