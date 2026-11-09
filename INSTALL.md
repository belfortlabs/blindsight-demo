# Installing

The demo runs on macOS 14 or newer on Apple Silicon, on Linux (x86-64 or ARM, a recent
distribution) and on Windows through WSL 2. The commands are the same everywhere.

## uv

Before the first run, and independent of the platform, you need uv, which sets up the Python
environments. Everything in the repository runs with `uv run`. The first `uv run` in a
directory creates its `.venv/` with a suitable Python and the packages from `pyproject.toml`,
and later runs reuse it. To install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal afterwards, or run the `source` line the installer printed, and try
`uv --version`.

Homebrew users can `brew install uv` instead. uv's
[installation guide](https://docs.astral.sh/uv/getting-started/installation/) lists the
other ways.

## macOS

On macOS, just two items need to be arranged: the C++20 compiler and the camera access.

### Compiler

The Belfort ML client builds with `clang++`, which is part of Apple's Command Line Tools:

```bash
xcode-select --install
```

If they are already installed the command tells you so. `clang++ --version` should then work.

### Camera access

`client.py` asks for camera access for your terminal app the first time. If you declined, you
can still allow it in System Settings > Privacy & Security > Camera.

## Linux and WSL 2

Linux and WSL 2 take a bit more work. The Belfort ML client needs a C++20 compiler with
`<format>`, enough memory to generate the key pair, and the system libraries for its Qt window,
and on WSL 2 the webcam has to be handed over from Windows. Each section below quotes the error
you get when that piece is missing.

### Compiler

The client uses `clang++`, or the compiler that `CXX` names; without `clang++` the first run
stops with `No such file or directory: 'clang++'`. GCC works too, but it has to be version 13
or newer: the encryption code includes `<format>`, which GCC's standard library only has from
13 on, and with GCC 11 or 12 the first run stops with
`fatal error: format: No such file or directory`. On Ubuntu 22.04, install GCC 13 from the
toolchain PPA and run `client.py` with `CXX=g++-13`:

```bash
sudo apt install software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-13
CXX=g++-13 uv run client.py --samples   # in eml/
```

Ubuntu 24.04 and newer ship GCC 13 or newer, so `CXX=g++` is enough there. A minimal
install, as in WSL, may not have the compiler at all. Install it once:

```bash
sudo apt install g++
CXX=g++ uv run client.py --samples   # in eml/
```

### Memory on WSL 2

If `client.py` stops with `harness exited with -9 before answering`, Linux ran out of memory
while generating the key pair (`dmesg` shows `Out of memory: Killed process ...
(personnet_clien)`). Give WSL more memory and swap in `%UserProfile%\.wslconfig` on the
Windows side: `memory` as much as Windows can spare, and a `swap` file as a backstop.

```ini
[wsl2]
memory=12GB
swap=16GB
```

If Windows cannot spare that much memory, use less and make the swap larger.

Then run `wsl --shutdown` in PowerShell and reopen the WSL terminal; `free -h` shows the new
limit. To see what the client needs on your machine, watch `VmHWM` (its peak) in
`/proc/<pid>/status` of the `personnet_client` process while it generates the keys. The
shutdown also stops a running deploy, so start `provider.py serve` again and use its new
client token.

### Qt libraries for the window

The window is Qt (PySide6), which needs some system libraries that a minimal Ubuntu, as in
WSL, does not have. Without them, `client.py` stops with
`Could not load the Qt platform plugin "xcb"`, or Qt warns about the `wayland` plugin or
missing `window-*-symbolic.svg` icons. Install them once:

```bash
sudo apt install libxcb-cursor0 libegl1 libgl1 libxkbcommon-x11-0 libxcb-icccm4 \
    libxcb-keysyms1 libxcb-shape0 libwayland-cursor0 adwaita-icon-theme
```

When a Wayland display is available, as in WSLg, the client uses Qt's Wayland platform with
its adwaita title bar; WSLg gives X11 windows a dated one. Otherwise it uses Qt's X11
platform (`xcb`). Set `QT_QPA_PLATFORM` or `QT_WAYLAND_DECORATION` to choose others.

The adwaita title bar shows only the buttons that the desktop settings portal names, and WSL
has no portal: the title bar then has no close, minimize or maximize button. To add them, and
optionally make the title bar dark, install the portal, restart WSL with `wsl --shutdown` in
PowerShell, and set the layout in a new WSL terminal:

```bash
sudo apt install dbus-user-session xdg-desktop-portal xdg-desktop-portal-gtk \
    gsettings-desktop-schemas dconf-service
# after wsl --shutdown, in a new WSL terminal:
gsettings set org.gnome.desktop.wm.preferences button-layout ':minimize,maximize,close'
gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'   # optional
```

### Webcam on WSL 2

WSL does not see the Windows cameras until the USB webcam is attached to it with
[usbipd-win](https://github.com/dorssel/usbipd-win). Install it once on Windows
(`winget install --exact dorssel.usbipd-win`), then, in an administrator PowerShell:

```powershell
usbipd list                          # find the webcam's BUSID
usbipd bind --busid <BUSID>          # once: share it
usbipd attach --wsl --busid <BUSID>  # after every reboot or replug
```

In WSL, `ls /dev/video*` then lists the camera, and `uv run client.py` opens its window on
the Windows desktop through WSLg. Keep to the small MJPG formats, the default: usbip cannot
carry the camera's larger or uncompressed (YUYV) streams intact, and those show up as a
corrupted picture. The model sees only a small centre crop, so nothing is lost. While
attached, the webcam is gone from Windows; `usbipd detach --busid <BUSID>` gives it back. A
camera built into the laptop is often USB inside and works the same way. If `/dev/video0` is
missing after the attach, load the webcam driver with `sudo modprobe uvcvideo`; if the WSL
kernel has no such module, update it with `wsl --update` in PowerShell.

If `/dev/video0` exists but the client stops with `Permission denied`, `ls -l /dev/video*`
shows which fix you need:

- Group `video`: add yourself to it with `sudo usermod -aG video $USER`. The new group
  applies only to a new login, and a new WSL terminal often joins the old one. Run
  `wsl --shutdown` in PowerShell, open a new WSL terminal, and attach the webcam again.
  `id` then lists `video`.
- Owner `root:root` with mode `crw-------`: udev did not set up the device, because WSL runs
  without systemd. Turn systemd on in `/etc/wsl.conf`, then run `wsl --shutdown` and attach
  the webcam again. For a quick test, `sudo chmod 666 /dev/video*` works until the next
  attach.

```ini
[boot]
systemd=true
```

A webcam on usbip can get stuck in the kernel. A process that waits for it is then in the `D`
state, and no signal stops it, not even `kill -9`. Only `wsl --shutdown` in PowerShell
clears it; then attach the webcam again. The client tries the camera in a child process
before it makes the keys, and stops with `the camera did not answer` if no frame comes in
time. If the camera gets stuck later, the status bar of the window shows the same message.

If step 3 of the client ends but the window does not open and `Window open` is not printed,
the client stops in the camera open or in the Qt start. Find which one, in `eml/`:

1. Test the camera without Qt. It prints `True True` when the camera delivers a frame.
   ```bash
   uv run python -c "import cv2; c = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2); print(c.isOpened(), c.read()[0])"
   ```
   If it does not finish, detach the webcam, attach it again, and do the test again.
2. If the camera works, start the client on X11 instead of Wayland:
   ```bash
   QT_QPA_PLATFORM=xcb uv run client.py
   ```
3. If it still does not open, start it with `PYTHONFAULTHANDLER=1 uv run client.py`. When it
   stops, run `pkill -ABRT -f client.py` in a second terminal. Python prints the stack of each
   thread, with the line where it waits.
