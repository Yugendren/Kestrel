# Branches and contributing

## Branches

- `main` is the stable development line. Anyone should use and build it.
- `upstream-main` is a mirror of upstream KytyPS5 history (the old fork `main`).
- `exp/...` and `meas/...` are experiment and measurement branches. They may break, and may be
  rebased or deleted. Do not build on them.
- `feat/...`, `fix/...` and `perf/...` hold feature work. They merge into `main` only after
  they are built, tested on Windows and Linux, and measured.

Kestrel keeps upstream history with the original authors.

## Contributing

1. Branch from `main`.
2. Make small, focused commits with clear messages.
3. Build on Windows and Linux (see [QUICKSTART.md](QUICKSTART.md)).
4. Run the tests:

   ```
   cmake --build <dir> --target kyty_tests
   ctest --test-dir <dir> --output-on-failure
   ```

5. For performance changes, show before and after numbers: same scene, same settings.
6. Make general fixes in the emulator. Make title-specific changes only as patch files.
7. When porting commits from other forks (GPL-2.0), keep the original authors.
8. Never commit game files, keys or firmware.

### Formatting

Code is formatted with clang-format through pre-commit:

```
python -m pip install pre-commit
python -m pre_commit install --install-hooks
```

## Reporting bugs and results

Open a GitHub issue on [Yugendren/Kestrel](https://github.com/Yugendren/Kestrel) with the output
of the launcher's report mode. See [QUICKSTART.md](QUICKSTART.md#reporting-results).
Do not attach game files.
