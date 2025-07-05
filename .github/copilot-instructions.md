This extension uses a precise development workflow:

  - During development, the extension should be configured with `--enable-ort-coverage`.
  - Any changes in source must be followed by `make clean install` *before* tests are run.
  - Tests must always be run with `make ort-test-coverage-html` to ensure:
      - They are executed in the correct environment.
      - Coverage information is generated in `./coverage.info` and the `./html/` is updated.
  - When working on code coverage, always refer to `./coverage.info` and not dumb analysis.

This extension uses a precise optimization workflow:

  - During optimization workflows, planning documents should be referenced:
    - If the current optimization workflow is unclear, ask which document to reference.
    - Documents must be kept up-to-date at each step of the process.