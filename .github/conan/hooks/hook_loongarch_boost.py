import os
from conan.errors import ConanException
from conan.tools.files import replace_in_file

def pre_build(conanfile, **kwargs):
    if (
        conanfile.name != "boost"
        or str(conanfile.version) != "1.83.0"
        or str(conanfile.settings.get_safe("arch")) != "loongarch64"
    ):
        return

    jamfile = os.path.join(
        conanfile.source_folder,
        "libs",
        "context",
        "build",
        "Jamfile.v2",
    )

    if not os.path.isfile(jamfile):
        raise ConanException(f"Could not locate Boost.Context Jamfile: {jamfile}")

    replace_in_file(
        conanfile,
        jamfile,
        "else if [ os.platform ] in ARM ARM64 { tmp = aapcs ; }",
        'else if [ os.platform ] = "ARM" { tmp = aapcs ; }\n'
        '    else if [ os.platform ] = "ARM64" { tmp = aapcs ; }',
    )

    recipe = type(conanfile)
    recipe._b2_address_model = property(lambda self: "64")
    recipe._b2_architecture = property(lambda self: "loongarch")
    recipe._b2_abi = property(lambda self: "sysv")

    conanfile.output.info("Patched Boost 1.83.0 Context ABI detection for LoongArch64")
