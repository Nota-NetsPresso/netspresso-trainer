# Contributing to this repository

## Install netspresso-trainer locally

TBD

## Install linter

First of all, you need to install `ruff` package to verify that you passed all conditions for formatting.

```
pip install ruff==0.0.287
```

### Apply linter before PR

Please run the ruff check with the following command:

```
ruff check src/netspresso_trainer
```

or run the linter with [`bash scripts/lint_check.sh`](./scripts/lint_check.sh).

If it shows some error, you should fix your errors to pass all conditions.
If there is no error output, feel free to post your PR 😊

### Auto-fix with fixable errors

```
ruff check src/netspresso_trainer --fix
```

If there is no remaining error, you are eligible to post your PR 😊

### Manual-fix with remaining errors

To fix these errors, contributors should understand the semantic meaning of the code. Please fix this manually until all errors are removed.

```
ruff check src/netspresso_trainer
```

should verify that your updates are minimally formatted for this repository.
