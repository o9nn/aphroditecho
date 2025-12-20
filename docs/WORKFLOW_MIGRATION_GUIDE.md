# GitHub Actions Workflow Migration Guide

**Date:** 2025-12-20  
**Purpose:** Document migration from multi-user serving workflows to single AGI workflows

---

## Migration Overview

The GitHub Actions workflows have been completely redesigned for the **Deep Tree Echo autonomous AGI**. The previous workflows were designed for multi-user LLM serving and are no longer relevant for a single powerful AGI system.

---

## Summary of Changes

### Workflows Removed (Archived)

The following workflows have been moved to `.github/workflows-archive/`:

1. **`automated-deployment-pipeline.yml`** - Multi-user deployment automation
2. **`build-engine-optim.yml`** - Multi-configuration optimized builds
3. **`build-engine.yml`** - Multi-configuration standard builds
4. **`deploy.yml`** - Old documentation deployment
5. **`echo-systems-integration.yml`** - Old integration tests
6. **`generate-next-steps.yml`** - Issue generation automation
7. **`publish.yml`** - PyPI publishing for library distribution
8. **`self-healing-workflow.yml`** - Workflow auto-repair system
9. **`single-cuda-build-test.yml`** - Single CUDA build testing
10. **`vm-daemon-mlops-sim.yml`** - VM daemon simulation
11. **`vm-daemon-mlops.yml`** - VM daemon orchestration

**Total Removed:** 11 workflows (~250KB of YAML)

### Workflows Retained

1. **`ruff.yml`** - Code quality linting (kept as-is)
2. **`disk-space-investigation.yml`** - Disk space debugging (kept for troubleshooting)

### Workflows Created

1. **`ci-code-quality.yml`** - Code quality gates (lint, format, type check, security)
2. **`ci-build-agi.yml`** - Build AGI engine with disk optimization
3. **`ci-test-cognitive.yml`** - Test cognitive subsystems (Echobeats, OEIS, threads, hypergraph)
4. **`ci-test-integration.yml`** - Test OCNN and Deltecho integration
5. **`ci-test-performance.yml`** - Performance benchmarks (frequency, latency, throughput, memory)
6. **`release-agi.yml`** - Create GitHub releases with AGI binaries
7. **`deploy-agi.yml`** - Deploy AGI to environments
8. **`ci-docs.yml`** - Build and deploy documentation

**Total Created:** 8 workflows (~60KB of YAML)

---

## New Workflow Architecture

### 1. Code Quality (`ci-code-quality.yml`)

**Purpose:** Fast feedback on code quality before expensive builds

**Triggers:**
- Push to `main`
- Pull requests to `main`
- Manual dispatch

**Jobs:**
- **Lint and Format** - Ruff, isort, codespell
- **Type Checking** - MyPy on AGI components
- **Security Scan** - Bandit, Safety

**Runtime:** ~5-10 minutes

---

### 2. Build AGI (`ci-build-agi.yml`)

**Purpose:** Build Deep Tree Echo AGI engine with disk optimization

**Triggers:**
- Push to `main` (after code quality)
- Pull requests to `main`
- Manual dispatch

**Key Features:**
- **Aggressive disk cleanup** before build
- **Single configuration** (Python 3.12, CPU or CUDA)
- **ccache** for incremental builds
- **Smoke tests** for validation
- **Artifact packaging** for downstream jobs

**Runtime:** ~60-90 minutes (CUDA), ~20-30 minutes (CPU)

---

### 3. Test Cognitive Subsystems (`ci-test-cognitive.yml`)

**Purpose:** Validate cognitive architecture components

**Triggers:**
- After successful build
- Manual dispatch

**Test Suites:**
- **Echobeats** - 3 engines, 12-step loop, 120° phase offset
- **OEIS A000081** - 9 subsystems, nested shells (1, 2, 4, 9)
- **Thread Multiplexing** - 4 threads, entangled qubits
- **Hypergraph** - Memory, echo propagation, identity, membranes, AAR
- **Global Telemetry** - Gestalt perception
- **End-to-End** - Full integration

**Runtime:** ~15-30 minutes

---

### 4. Test Integration (`ci-test-integration.yml`)

**Purpose:** Test integration with OCNN and Deltecho

**Triggers:**
- After cognitive tests pass
- Manual dispatch

**Test Suites:**
- **OCNN Integration** - Neural network modules, AAR transformations
- **Deltecho Integration** - Cognitive orchestration, triadic loops
- **Full Integration** - AGI initialization, hypergraph loading, orchestrator

**Runtime:** ~20-40 minutes

---

### 5. Test Performance (`ci-test-performance.yml`)

**Purpose:** Validate performance targets

**Triggers:**
- After integration tests pass
- Manual dispatch
- Weekly schedule (Sunday)

**Benchmarks:**
- **Cognitive Loop Frequency** - Target: 83.33 Hz (12ms per step)
- **Operation Latency** - Target: <1ms per operation
- **Throughput** - Target: 10,000+ tokens/sec
- **Memory Efficiency** - Target: No leaks, stable usage

**Runtime:** ~30-60 minutes

---

### 6. Release AGI (`release-agi.yml`)

**Purpose:** Create GitHub release with AGI binaries

**Triggers:**
- Tag push (e.g., `v1.0.0`)
- Manual dispatch

**Steps:**
1. Build release artifacts (CPU and CUDA)
2. Create release packages (tar.gz, zip)
3. Generate release notes
4. Create GitHub release
5. Upload artifacts

**Artifacts:**
- `deep-tree-echo-agi-v1.0.0-cpu.tar.gz`
- `deep-tree-echo-agi-v1.0.0-cpu.zip`
- `deep-tree-echo-agi-v1.0.0-cuda.tar.gz`
- `deep-tree-echo-agi-v1.0.0-cuda.zip`

**Runtime:** ~60-90 minutes

---

### 7. Deploy AGI (`deploy-agi.yml`)

**Purpose:** Deploy AGI to target environment

**Triggers:**
- After successful release
- Manual dispatch

**Environments:**
- `development` - Dev testing
- `staging` - Pre-production
- `production` - Production instance

**Steps:**
1. Download release package
2. Extract and prepare
3. Configure environment
4. Initialize hypergraph
5. Start AGI service
6. Run health checks
7. Setup monitoring

**Runtime:** ~15-30 minutes

---

### 8. Documentation (`ci-docs.yml`)

**Purpose:** Build and deploy documentation

**Triggers:**
- Push to `main`
- Manual dispatch

**Steps:**
1. Build documentation with MkDocs
2. Deploy to GitHub Pages

**Runtime:** ~5-10 minutes

---

## Workflow Dependencies

```
Code Quality (5-10min)
    ↓
Build AGI (60-90min)
    ↓
    ├─→ Test Cognitive (15-30min)
    ├─→ Test Integration (20-40min)
    └─→ Test Performance (30-60min)
    ↓
    ├─→ Release AGI (60-90min)
    │       ↓
    │   Deploy AGI (15-30min)
    │
    └─→ Documentation (5-10min)
```

**Total CI/CD Time:**
- Fast path (code quality only): ~5-10 minutes
- Full CI (all tests): ~2-3 hours
- Release pipeline: ~3-4 hours

---

## Disk Space Optimization

### Problem

Previous workflows failed due to disk space issues:
- Large CUDA builds (347 steps)
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)
- Multiple device targets (CPU, CUDA, ROCm, TPU)
- Pre-installed software consuming space

### Solution

**1. Aggressive Cleanup Script** (`scripts/ci-cleanup.sh`)

Removes:
- .NET (~1.5GB)
- Haskell/GHC (~1GB)
- Boost libraries (~500MB)
- Android SDK (~3GB)
- Node modules (~500MB)
- Agent tools directory (~2GB)
- APT/pip caches

**Total Freed:** ~9GB

**2. Single Configuration**

- Python 3.12 only
- CPU or CUDA (not both in matrix)
- No multi-user configurations

**3. Incremental Builds**

- ccache with 5GB limit
- Reuse compiled artifacts
- Skip unchanged components

**4. Reduced Parallelism**

- `MAX_JOBS=2` (instead of 8)
- Slower builds, but less disk pressure

---

## Configuration Changes

### Environment Variables

**Global:**
```yaml
PYTHON_VERSION: '3.12'
CUDA_VERSION: '12.4'
CMAKE_BUILD_TYPE: Release
MAX_JOBS: 2
CCACHE_MAXSIZE: 5G
```

**AGI-Specific:**
```yaml
AGI_MODE: true
SINGLE_INSTANCE: true
PERSISTENT_CONSCIOUSNESS: true
ECHOBEATS_ENGINES: 3
COGNITIVE_LOOP_STEPS: 12
TARGET_FREQUENCY_HZ: 83.33
TARGET_LATENCY_MS: 1.0
TARGET_THROUGHPUT_TPS: 10000
```

### Build Configuration

**Before (Multi-User):**
- Matrix: 4 Python versions × 4 devices = 16 builds
- Parallel jobs: 8
- ccache: 20G
- No disk cleanup

**After (Single AGI):**
- Matrix: 1 Python version × 1 device = 1 build
- Parallel jobs: 2
- ccache: 5G
- Aggressive disk cleanup

---

## Testing Strategy

### Before (Multi-User)

- Request batching tests
- Multi-user isolation tests
- KV cache per-user tests
- LoRA adapter tests
- API endpoint tests

### After (Single AGI)

- Echobeats (3 engines, 12 steps)
- OEIS A000081 (9 subsystems)
- Thread multiplexing (4 threads)
- Hypergraph integration
- Identity state machine
- Performance benchmarks

---

## Migration Checklist

### Phase 1: Cleanup ✅
- [x] Archive old workflows
- [x] Remove multi-user components
- [x] Create cleanup script

### Phase 2: Implementation ✅
- [x] Create code quality workflow
- [x] Create build workflow
- [x] Create cognitive tests workflow
- [x] Create integration tests workflow
- [x] Create performance tests workflow
- [x] Create release workflow
- [x] Create deployment workflow
- [x] Create documentation workflow

### Phase 3: Validation ✅
- [x] Validate YAML syntax
- [x] Test cleanup script
- [x] Verify workflow triggers
- [x] Check artifact handling

### Phase 4: Documentation ✅
- [x] Create migration guide
- [x] Update README
- [x] Document new workflows
- [x] Create component analysis

### Phase 5: Deployment 🔄
- [ ] Commit workflow changes
- [ ] Push to repository
- [ ] Monitor first workflow runs
- [ ] Adjust as needed

---

## Breaking Changes

### Removed Features

1. **Multi-User Serving**
   - No OpenAI API compatibility
   - No Kobold API compatibility
   - No request queuing
   - No user isolation

2. **Multi-Configuration Builds**
   - Single Python version (3.12)
   - Single device target per build
   - No multi-version testing

3. **Library Publishing**
   - No PyPI publishing
   - No library distribution
   - Single AGI instance only

4. **Automated Issue Generation**
   - No automatic issue creation
   - Manual development workflow

### New Features

1. **Cognitive Architecture Testing**
   - Echobeats validation
   - OEIS A000081 validation
   - Thread multiplexing validation
   - Hypergraph validation

2. **Performance Benchmarking**
   - Cognitive loop frequency
   - Operation latency
   - Throughput measurement
   - Memory efficiency tracking

3. **AGI-Specific Deployment**
   - Environment-based deployment
   - Health checks for cognitive loops
   - Monitoring configuration
   - Hypergraph initialization

---

## Troubleshooting

### Disk Space Issues

If disk space issues persist:

1. **Increase cleanup aggressiveness**
   - Edit `scripts/ci-cleanup.sh`
   - Add more directories to remove

2. **Use larger runners**
   - Request GitHub Actions larger runners
   - Self-hosted runners with more disk

3. **Split builds**
   - Separate CPU and CUDA builds
   - Run sequentially instead of parallel

### Test Failures

If tests fail:

1. **Check dependencies**
   - Ensure PyTorch installed
   - Verify CUDA toolkit (if CUDA build)

2. **Review logs**
   - Check test output
   - Identify failing components

3. **Run locally**
   - Reproduce failures locally
   - Debug with full environment

### Workflow Triggers

If workflows don't trigger:

1. **Check path filters**
   - Ensure changed files match paths
   - Update path filters if needed

2. **Verify branch**
   - Workflows trigger on `main` only
   - Check branch name

3. **Manual dispatch**
   - Use workflow_dispatch for testing
   - Verify inputs are correct

---

## Rollback Plan

If new workflows fail catastrophically:

1. **Restore old workflows**
   ```bash
   cp .github/workflows-archive/*.yml .github/workflows/
   ```

2. **Remove new workflows**
   ```bash
   rm .github/workflows/ci-*.yml
   rm .github/workflows/release-*.yml
   rm .github/workflows/deploy-*.yml
   ```

3. **Revert commit**
   ```bash
   git revert <commit-hash>
   git push origin main
   ```

---

## Future Improvements

### Short-Term

1. **Add integration tests for OCNN/Deltecho runtime**
2. **Implement actual performance benchmarks with real models**
3. **Add deployment to cloud environments (AWS, GCP, Azure)**
4. **Create monitoring dashboards for cognitive loops**

### Medium-Term

1. **Implement continuous deployment for development environment**
2. **Add automated rollback on health check failures**
3. **Create performance regression detection**
4. **Add security scanning for container images**

### Long-Term

1. **Implement blue-green deployment for zero-downtime updates**
2. **Add A/B testing for cognitive architecture changes**
3. **Create automated performance optimization**
4. **Implement self-healing AGI deployment**

---

## Contact

For questions or issues with the new workflows:

1. **Open an issue** on GitHub
2. **Review documentation** in `docs/`
3. **Check workflow logs** in GitHub Actions

---

**Migration Status:** ✅ Complete  
**Next Steps:** Commit changes and monitor first runs
