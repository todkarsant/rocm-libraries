# hipBLASLt just-in-time (JIT) GEMM generation

This source guide describes just-in-time (JIT) kernel generation for general
matrix multiplication (GEMM) in hipBLASLt. It is written for hipBLASLt and
TensileLite contributors and integration developers, and it is maintained under
the existing code-owner rules in [.github/CODEOWNERS](../../.github/CODEOWNERS).
It is not a released application programming interface (API) or support
statement. Release-document integration is undecided.

It describes what the code implements today. "Implemented" means present in the
source, not released or approved as product naming. A JIT backend is the
generator that turns a request into kernel sources; this page describes the
parts of hipBLASLt around any backend. The
[JIT test guide](clients/tests/jit/README.md) covers building and running the
tests.

## Summary

hipBLASLt runs a GEMM with a kernel from its pre-tuned library, so a problem
that the library serves poorly, or not at all, has no better kernel available.
JIT generation produces kernels for a problem when they are needed.
A comgr code-object builder, a TensileLite source bundle reader and the JIT
solution library are implemented. They build generated kernel sources into
code objects and publish them as TensileLite solutions into a persistent JIT
solution library on disk, whose solutions a later process loads without
building again.

## Current behavior

### Components

The interfaces are in `library/src/amd_detail/hipblaslt-jit-component.hpp`.

An `Artifact` is one file of one kind: `Assembly` or `Hip` source, a
`Relocatable` (compiled and not linked) or an `Executable` (a linked code
object). A `SourceCodeBundle` is the main kernel, helpers and headers that
become one executable, with the build inputs: the device's `Target`, the
code-object version, the `ContractionSolution`s the executable runs, best
first, and any other kernel names, such as helpers', the executable must
define. It is not a TensileLite library entry.
`CodeObjectBuilder::build(bundle, kind, out)` builds that bundle with comgr
into exactly one `Artifact`: an `Executable` for `BuildKind::Executable`, or
the main kernel alone as a `Relocatable` for `BuildKind::Relocatable`, which
is how a helper is built.

A `TensileliteAssembly` is a TensileLite kernel's `Assembly` with its `Target`
and the `ContractionSolution`s it implements. `TensileliteAssembly::make`
assembles the kernel alone into a `Relocatable`, reads the solution state
TensileLite keeps in `custom.config` of the kernel's `.amdgpu_metadata` from
that object's metadata note (`code_object::metadataNote`), and builds the
solution with `TensileLite::solutionFromState`. The kernel name and the
kernel-argument versions are the solution's; the target is a field because a
solution states its target only as a hardware predicate. An `Executable` build
accepts that `Relocatable` as its main kernel and links it without assembling
again. `readTensileSourceBundle` reads a source bundle directory and does this
for its assembly, so the bundle's main kernel is the `Relocatable` and its
`solutions` are the kernel's. A HIP main kernel stays a plain `Artifact`, and
its backend builds the bundle's `ContractionSolution`s itself.

A `Target` holds the TensileLite `AMDGPU::Processor`, an `Any`, `On` or `Off`
setting for each of `sramecc` and `xnack`, and the device's ASIC revision.
`Target::fromTargetId` parses a target ID, such as
`hipDeviceProp_t::gcnArchName` or the ID in an assembly `.amdgcn_target`
directive, and rejects an unknown processor, including `gfx000`, and any other
feature. Strings come only from `processorName()`, `targetId()`, `isaName()`
(the comgr ISA name) and `libraryArch()` (the GEMM library subtree, `gfx1250v0`
for a gfx1250 A0 part), so library directory names and `cache-key.msgpack`
fields use only these strings.

The interfaces are for compiled-in implementations
and do not establish a stable external plugin application binary interface (ABI).

The implementations are:

- Code-object builder: `makeComgrBuilder()`; see
  [building generated sources](#building-generated-sources).
- Support check: `supportingSolutions`; see
  [loading a published solution](#loading-a-published-solution).
- Solution store: the JIT solution library, in `hipblaslt-jit-library.cpp`,
  with `hipblaslt-jit-msgpack.cpp` writing its own documents and
  `hipblaslt-jit-fs.cpp` providing the directory checks, file lock and atomic
  replacement; see [persistent solution library](#persistent-solution-library).

### Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. A disabled build compiles and
exports no JIT entry points. The enabled build requires the host library, ROCm
and ROCm's `amd_comgr` CMake package, which only a JIT build links. comgr
compiles HIP sources against the host's C and C++ standard library headers,
so those must be installed where JIT runs. From the repository root:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
```

`GPU_TARGETS` is `gfx90a`, `gfx942` or `gfx950`. The command above uses
`gfx950`. The [JIT test guide](clients/tests/jit/README.md) lists the test
targets and the validation commands.

### Persistent solution library

The JIT solution library keeps generated solutions on disk so that later
processes run them without generating again. It is the only cache.
`JitLibrary::lookup` returns up to a requested number of the indices
published for exactly a problem, in the order they were first published, and
`findTopSolutions` returns those solutions for the query that `beginQuery`
sets on the calling thread.

**Location and permissions.** `JitLibrary` takes its root from the caller.
hipBLASLt creates the root and each directory it uses below it when they are
missing, with mode 0700 on Linux. On Linux it refuses the root, and each
directory it uses below the root, when the path is a symbolic link or not a directory, is
owned by another user, or is writable by group or others; a readable directory
such as 0755 is accepted. Windows checks only for reparse points and
non-directories. There is no setting that accepts a shared, writable
directory. A refused root, a privileged (setuid or setgid) process, a build
that reads YAML libraries and a build without a git revision disable the
library, and each use then fails with `JIT solution library disabled: <reason>`.

**Layout.** Under a `v1` schema directory, each hipBLASLt build has its own
directory, named by the git revision `hipblasLtGetGitRevision` reports. A build
reads and writes only its own directory, so it never loads kernels that another
build published, even when that build packs kernel arguments differently.
Within it, each device and backend has its own directory. The directory name is a filesystem-safe literal of the ISA, target
ID, library architecture, wavefront size, backend id and backend version. It
does not include the comgr path or the ROCm path. A query opens one slice of
that directory. The slice key is TensileLite's lazy placeholder grain: the
operation identifier, the datatypes, the scale suffix (`_SAB` or `_SABV`, and
MX block and type) and the epilogue flags that change the kernel family. The
slice is a standard lazy TensileLite library that the stock loader reads: a
master file, the index mapping, and one entry (`.dat`) and code object (`.co`)
per published executable. Other slices stay unread.
The directory also holds `helpers/`, one relocatable per helper
kernel, named for the helper kernel's symbol. A solution's code object is the
fully linked executable, produced by linking the main kernel with those helper
relocatables. The files the library reads and writes are MessagePack.
TensileLite's writer (`MessagePackWriteLibraryData` and
`MessagePackWriteLibraryMapping`) writes the master, the mapping and the
entries from library objects built in C++. The JIT library's own documents are
`cache-key.msgpack`, the index allocator and the index-to-slice map. For
example:

```text
<root>/
└── v1/
    └── <git revision>/                    one directory per hipBLASLt build
        ├── allocator.dat                  next free solution index
        ├── lock                           publication lock for this build
        └── gfx950_..._<backend>_<version>/ one directory per device and backend
            ├── cache-key.msgpack
            ├── indices.msgpack            solution index to slice key
            ├── helpers/
            │   └── <helper kernel>.o      relocatable linked into a solution
            ├── slices/<slice key>/
            │   ├── TensileLibrary_lazy_gfx950.dat
            │   ├── TensileLiteLibrary_lazy_gfx950_Mapping.dat
            │   ├── k/<kernels>/t/<problem>/s/<sizes>.dat
            │   └── k/<kernels>/t/<problem>/s/<sizes>.co
            └── staging/                   temporary files before rename
```

An entry name is a filesystem-safe literal of the kernels, the problem type and
the sizes. An entry holds the supported solutions published with one
executable, each under its own allocated index, and that executable beside it
as the code object. Publishers of that same identity choose the same files,
and a solution already in the entry keeps its index. Each solution has its own
master row, `And(SizeEqual(M), SizeEqual(N), SizeEqual(batch), SizeEqual(K),
<the solution's problem predicate>)`, pointing to a placeholder for the entry,
so a lookup matches only the exact sizes within the full ProblemType. The solution's own
hardware, problem and task predicates, including its workspace requirement,
still run on every match. An entry this build cannot read fails the lookup
instead of being skipped, because only this build wrote it. A different comgr
path or ROCm path does not reject the directory.

**Cache key.** `cache-key.msgpack` records:

| Field | Contents |
| --- | --- |
| `target_id`, `isa`, `library_arch`, `wavefront_size` | Full target ID with features (for example `gfx950:sramecc+:xnack-`), ISA, TensileLite library architecture and wavefront size |
| `backend_id`, `backend_version` | Backend identifier and integer version |
| `comgr` | comgr version, and on Linux the path, size and modification time of the loaded comgr library |
| `code_object_version` | The code-object version that the generator and the builder use (4) |
| `rocm_path` | The ROCm path that the builder passes to comgr |
| `compiler_environment` | `HIP_PATH`, `LLVM_PATH`, and every set `AMD_COMGR_*` variable, including `AMD_COMGR_CACHE`, `AMD_COMGR_CACHE_DIR`, `AMD_COMGR_CACHE_POLICY` and `AMD_COMGR_USE_VFS`. Logging, `SAVE_TEMPS`, `SAVE_LLVM_TEMPS` and `TIME_STATISTICS` are omitted because they do not change code-object bytes |
| `schema` | The library schema version (1). A different integer does not reject the directory. |

A process uses the directory named for its device and backend. A present device
or backend field that disagrees with that directory fails the lookup or
publication. The comgr identity, the ROCm path, the schema integer and the
code-object version are recorded and do not. To resolve an index it has not
looked up, a `JitLibrary` constructed with a `DeviceKey` searches the
directories for its device, so a later process can run the solution without
configuring the backend that generated it.

**Indices.** JIT solutions use solution indices from `jitIndexBase` (2^30) up
to, but not including, `jitIndexEnd` (0x60000000): 2^29 indices. Indices from
0x60000000 to `INT32_MAX` are left to runtime user kernel registration and are
never JIT indices. `allocator.dat` holds the next index, so indices stay
unique across every key directory of a build and are never reused while that
build's directory exists. Publication fails once the range is exhausted,
without allocating `jitIndexEnd`.

**Publication and refresh.** A publisher holds `lock` while it allocates
indices and writes, in this order: `allocator.dat`, the code objects, the
entries, the mapping and the master. Each file is written under `staging/` and
renamed into place, so readers never see a partial file. A process that stops
at any point leaves a library that loads, whose master refers only to complete
entries; the next publisher reuses or overwrites what it left. A publisher
waits at most 15 seconds for `lock`. That is a ceiling, not a typical wait: an
uncontended publish takes the lock at once, and a waiting publisher polls at
intervals growing from 1 ms to 100 ms and takes the lock as soon as it is
free. The lock covers only another publisher's file writes; builds and helper
compiles run before a publisher takes it. A process that exits releases its
lock. A publisher that times out fails and publishes nothing. Republishing a
published solution writes nothing and returns its index. Each lookup reloads
the master when another process has replaced the master or mapping file, and
resolving an unknown index reloads it too. A solution already resolved from an
earlier master stays valid.

**Clearing.** hipBLASLt never deletes entries. To clear the library, delete
the root directory, or one key directory, while no process is using it.

### Building generated sources

Generators emit assembly or HIP source plus metadata only; they do not assemble,
link or bundle. `makeComgrBuilder()` returns the `CodeObjectBuilder` that builds
code objects in process through AMD comgr (`hipblaslt-jit-builder.cpp` and
`hipblaslt-jit-code-object.cpp`). It uses three comgr actions:

- Assembly: `AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE`. The main kernel's
  `.amdgcn_target` directive must name the device's processor and only
  features the device has; it is then rewritten to the device's full target
  ID. The main kernel's declared wavefront size is the one the assembler uses. An
  `Executable` build also accepts a main kernel already assembled into a
  `Relocatable`.
- HIP source: `AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE` with
  `--rocm-path` and a `-cuid` taken from the source name. A HIP helper
  compiles to its own `Relocatable`. `JitLibrary::acquireHelper` stores that
  relocatable as a file in the cache-key directory, named for the helper
  kernel's symbol, because a source file name such as a TensileLite
  `Kernels.cpp` can differ per solution. A hit returns it and does not compile.
  A miss compiles it once and stores it under the publish lock. A bundle's
  helpers are HIP sources, which an `Executable` build compiles, or
  relocatables, which it links as they are; the build keeps no second cache.
  comgr's `AMD_COMGR_CACHE` stays comgr's; it cannot be stored in the library.
  The ROCm path is `HIP_PATH` when set, otherwise the prefix of the loaded HIP
  runtime.
- Link: `AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE` links the main kernel
  and its helpers into one `Executable`, with `-Xlinker --build-id=sha1`. The
  solution library stores that executable.

The generator and the builder use the same code-object version, which
`SourceCodeBundle::codeObjectVersion` carries (4 by default). The output is a
raw, uncompressed executable code object. comgr cannot bundle or compress it,
and `hipModuleLoadData` accepts raw executable and linkable format (ELF)
objects. A generator needs no offload bundler.

After linking, the builder reads the executable's metadata and checks that its
instruction set architecture (ISA) is the device's, that its code-object
version is the bundle's, and that it defines the kernel of every solution in
`SourceCodeBundle::solutions` and every name in `SourceCodeBundle::kernelNames`.
The builder writes no file. A failure's
message carries the first error line of the comgr log, and `Status::log`
carries the full log. A caller that owns a scratch directory passes the status
to `keepBuildLog`, which appends the log to `comgr.log` there and names that
file in the message.

### Loading a published solution

`JitLibrary` is a TensileLite `SolutionLibrary`. A published solution launches
through the TensileLite solution adapter the same way Equality and Origami
solutions do. The stored record is a `ContractionSolution` with the code
object, kernel name, `ContractionSolution::ProblemType`, and the predicates
those libraries already expect. Nothing else holds a published solution in
process.

`supportingSolutions`, in `hipblaslt-jit-library.hpp`, decides which of a
bundle's solutions serve a problem before they are published. TensileLite's
`selectionPredicate` evaluates each solution's problem, task and hardware
predicates, with the workspace limit as the problem's workspace, and the
solution's `requiredWorkspaceSize` must fit that limit. It keeps the bundle's
order, best first. A backend contributes one `SourceCodeBundle`. `publish`
takes the executable built from it and the solutions `supportingSolutions`
kept, and adds them as one entry.

Each solution of an entry has its own `Problem` row. Their hardware and
problem predicates decide which problems each one serves, so a lookup lists,
in index order, every solution of a matching entry that serves the problem,
and nothing for a problem none of them was generated for.

### Source bundle format

A source bundle is one kernel's sources stored as a directory.
`source_bundle::readSourceBundle`, in `hipblaslt-jit-source-bundle.hpp`, reads
the sources into a `SourceCodeBundle` and requires one main kernel. The caller
sets the target and solutions, which the directory does not hold.
`readTensileSourceBundle` reads the solutions from the assembly's
`custom.config`, as above. A bundle with a `library/` entry is rejected,
because a kernel's solutions come from the kernel itself.

| Path | Contents |
| --- | --- |
| `sources/*.s` | The assembly main kernel, at most one |
| `sources/*.hip`, `sources/*.cpp` | The HIP main kernel when there is no assembly, otherwise HIP helpers |
| Other files in `sources/` | Headers |
| `manifest.json` | A JavaScript Object Notation (JSON) provenance record that hipBLASLt does not read |

Artifact paths must be relative and stay inside the bundle, including through
symbolic links, and `sources/` may hold only regular files. The reader bounds
the file count (1024), each file (64 MiB) and the sources in total (256 MiB).
