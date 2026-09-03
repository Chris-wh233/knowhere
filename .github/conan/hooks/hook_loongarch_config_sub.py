import os
import shutil
import subprocess

from conan.errors import ConanException


def pre_build(conanfile, **kwargs):
    if conanfile.name != "libiberty" or str(conanfile.settings.get_safe("arch")) != "loongarch64":
        return

    try:
        automake_libdir = subprocess.run(
            ["automake", "--print-libdir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConanException("Could not locate Automake's GNU config files") from error

    for filename in ("config.guess", "config.sub"):
        source = os.path.join(automake_libdir, filename)
        destination = os.path.join(conanfile.source_folder, filename)
        if not os.path.isfile(source) or not os.path.isfile(destination):
            raise ConanException(f"Could not update libiberty's {filename}")
        shutil.copy2(source, destination)

    conanfile.output.info("Updated libiberty GNU config files for LoongArch64")
