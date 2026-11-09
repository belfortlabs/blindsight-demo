"""The Belfort ML model provider: compile PersonNet, then serve it to one
Belfort ML client.

    uv run provider.py compile   # compile the model on Belfort
    uv run provider.py serve --dev   # development only: a GPU deploy whose keys the gateway makes
    uv run provider.py serve     # mint a client token, wait for the client's keys, start a GPU deploy
    uv run provider.py delete    # delete the compiled model

compile and serve are safe to run again: a model already compiled from the
same MLIR and options is reused, not compiled again. The MLIR changes when the
weights or the calibration samples do; a different MLIR or options under the
same name is refused, so delete the model first.

serve --dev is developer mode (belfort-ml 0.2.5 or later): the deploy opens
its own key set, the gateway chooses its secret seed and the GPU generates the
evaluation keys from it, so the client uploads nothing. The seed is handed to
client.py like the token, so it needs no option. The gateway knows the seed, so
it can decrypt everything the deploy serves: never use it with real data.
"""

import hashlib
import sys
import termios
from datetime import datetime, timedelta, timezone
from pathlib import Path

import belfort_ml
import trained_model

# The compiled model's name on the gateway; names take letters, digits and single underscores.
NAME = trained_model.MODEL
# client.py on this machine reads the client token from here; it exists while serve runs.
# resolve(): the same file whichever directory provider.py and client.py are started from.
TOKEN_FILE = Path(__file__).resolve().parent / ".client-token"


def compiled_model(provider: belfort_ml.Provider) -> belfort_ml.Model:
    images, _ = trained_model.calibration_samples()
    calibration = list(images.split(1))
    artifacts = belfort_ml.compile(
        trained_model.load_model(), calibration[0], NAME, trained_model.HERE / "build", calibration_inputs=calibration
    )
    try:
        return provider.models.get(
            name=NAME, create_from=artifacts, timeout=1800, use_composite_relu=True, level_zero_encryption=True
        )
    except belfort_ml.ConflictError as exc:
        if exc.code != "name_taken":
            raise
        sys.exit(f"{exc}\nThe weights, samples, options or the Belfort release changed since {NAME} was "
                 f"compiled: run `uv run provider.py delete`, then compile again.")


def hand_over(token: str, seed: str | None = None) -> None:
    """Write the client token, and a developer-mode seed, where client.py looks for them."""
    TOKEN_FILE.touch(mode=0o600)
    TOKEN_FILE.write_text(token)
    export = f"  export BELFORT_CLIENT_TOKEN={token}\n"
    if seed is not None:
        # The same file name as client.py's seed_file(): one per client token.
        seed_file = TOKEN_FILE.parent / f".secret-seed-{hashlib.sha256(token.encode()).hexdigest()[:12]}"
        seed_file.touch(mode=0o600)
        seed_file.write_text(seed)
        export += f"  export BELFORT_CLIENT_SECRET_SEED={seed}\n"
    print(f"\nStart client.py now. On this machine it reads the client token from {TOKEN_FILE.name}.\n"
          f"On another machine, run this before client.py:\n{export}", flush=True)


def serve(model: belfort_ml.Model, dev: bool = False) -> None:
    TOKEN_FILE.unlink(missing_ok=True)  # a killed earlier run may have left one
    try:
        if dev:
            token = None
        else:
            # ISO 8601 basic format in UTC, e.g. client_20261017T141530Z: sorts in time order.
            token = model.create_client(label=f"client_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
            hand_over(token.token)
            # The GPU bills from here, so start it only once the client's keys are in.
            model.wait_for_keys(token)
            print("Keys uploaded. Starting the deploy; Ctrl-C stops it.", flush=True)
        with model.deploy(key_set=token, max_lifetime=timedelta(minutes=30), timeout=1800,
                          dev_mode=dev) as deploy:
            if dev:
                # The gateway chose the seed and minted a client token for this deploy.
                assert deploy.client_token is not None
                hand_over(deploy.client_token, deploy.secret_seed)
            # Drop keys pressed while the deploy was starting, so a stray Enter does not stop it at once.
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            input("Deploy ready. Press Enter to stop it...")
    finally:
        TOKEN_FILE.unlink(missing_ok=True)


def delete(provider: belfort_ml.Provider) -> None:
    try:
        provider.models.get(name=NAME).delete()
    except belfort_ml.NotFoundError:
        print(f"no model named {NAME}")
    else:
        print(f"deleted {NAME}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    dev = sys.argv[2:] == ["--dev"]
    if command not in ("compile", "serve", "delete") or (sys.argv[2:] and not (dev and command == "serve")):
        sys.exit(__doc__)
    with belfort_ml.Provider() as provider:  # reads BELFORT_API_KEY
        if command == "delete":
            delete(provider)
        else:
            model = compiled_model(provider)
            if command == "compile":
                print(f"{NAME} is compiled ({model.id})")
            else:
                serve(model, dev)
