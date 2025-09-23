#!/usr/bin/env python3
"""
Self-Healing System for Aphrodite Engine
Monitors system health and automatically creates issues for blocking errors
"""

import sys
import json
import logging
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Import GitHub API if available
try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    print("Warning: PyGithub not available. Install with: pip install pygithub")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorDetector:
    """Detects various types of blocking errors in the repository"""
    
    def __init__(self, repo_path: str = ".", github_repo=None):
        self.repo_path = Path(repo_path)
        self.github_repo = github_repo
        
    def detect_build_failures(self) -> List[Dict]:
        """Detect build-related failures"""
        errors = []
        
        # Check for build artifacts that indicate failures
        build_dirs = ['build', 'dist', '.cmake_build']
        for build_dir in build_dirs:
            build_path = self.repo_path / build_dir
            if build_path.exists():
                # Check for error logs
                error_files = list(build_path.rglob('*error*')) + list(build_path.rglob('*Error*'))
                if error_files:
                    errors.append({
                        'type': 'build_failure',
                        'location': str(build_path),
                        'error_files': [str(f) for f in error_files[:5]],  # Limit to 5 files
                        'severity': 'high'
                    })
        
        # Check CMake cache for configuration issues
        cmake_cache = self.repo_path / 'CMakeCache.txt'
        if cmake_cache.exists():
            try:
                with open(cmake_cache, 'r') as f:
                    content = f.read()
                    if 'ERROR' in content or 'FAILED' in content:
                        errors.append({
                            'type': 'cmake_configuration_error',
                            'location': str(cmake_cache),
                            'severity': 'medium'
                        })
            except Exception as e:
                logger.warning(f"Failed to read CMake cache: {e}")
        
        return errors
    
    def detect_test_failures(self) -> List[Dict]:
        """Detect test-related failures"""
        errors = []
        
        # Check for pytest artifacts
        pytest_cache = self.repo_path / '.pytest_cache'
        if pytest_cache.exists():
            lastfailed = pytest_cache / 'v' / 'cache' / 'lastfailed'
            if lastfailed.exists():
                try:
                    with open(lastfailed, 'r') as f:
                        failed_tests = json.load(f)
                        if failed_tests:
                            errors.append({
                                'type': 'test_failure',
                                'location': str(lastfailed),
                                'failed_tests': list(failed_tests.keys())[:10],  # Limit to 10 tests
                                'severity': 'medium'
                            })
                except Exception as e:
                    logger.warning(f"Failed to read pytest lastfailed: {e}")
        
        # Check for core dump files
        core_dumps = list(self.repo_path.rglob('core.*')) + list(self.repo_path.rglob('*.core'))
        if core_dumps:
            errors.append({
                'type': 'segmentation_fault',
                'location': 'multiple',
                'core_dumps': [str(f) for f in core_dumps[:3]],  # Limit to 3 files
                'severity': 'critical'
            })
        
        return errors
    
    def detect_import_errors(self) -> List[Dict]:
        """Detect Python import-related errors"""
        errors = []
        
        # Try importing core modules
        import_tests = [
            ('aphrodite', 'Core Aphrodite module'),
            ('aphrodite.engine', 'Aphrodite Engine'),
            ('aphrodite.endpoints', 'API Endpoints'),
        ]
        
        for module, description in import_tests:
            try:
                __import__(module)
            except ImportError as e:
                errors.append({
                    'type': 'import_error',
                    'module': module,
                    'description': description,
                    'error': str(e),
                    'severity': 'high'
                })
            except Exception as e:
                errors.append({
                    'type': 'module_error',
                    'module': module,
                    'description': description,
                    'error': str(e),
                    'severity': 'medium'
                })
        
        return errors
    
    def detect_deep_tree_echo_errors(self) -> List[Dict]:
        """Detect Deep Tree Echo specific errors"""
        errors = []
        
        # Check for Echo system directories
        echo_dirs = ['echo.kern', 'echo.self', 'echo.dash', 'echo.dream']
        missing_dirs = []
        
        for echo_dir in echo_dirs:
            echo_path = self.repo_path / echo_dir
            if not echo_path.exists():
                missing_dirs.append(echo_dir)
        
        if missing_dirs:
            errors.append({
                'type': 'missing_echo_components',
                'missing_directories': missing_dirs,
                'severity': 'medium'
            })
        
        # Check for DTESN processor errors
        dtesn_files = list(self.repo_path.rglob('*dtesn*.py'))
        for dtesn_file in dtesn_files:
            try:
                with open(dtesn_file, 'r') as f:
                    content = f.read()
                    if 'ERROR' in content or 'FAILED' in content or 'Exception' in content:
                        errors.append({
                            'type': 'dtesn_processor_error',
                            'location': str(dtesn_file),
                            'severity': 'high'
                        })
            except Exception as e:
                logger.warning(f"Failed to check DTESN file {dtesn_file}: {e}")
        
        return errors
    
    def detect_all_errors(self) -> Dict[str, List[Dict]]:
        """Detect all types of errors"""
        all_errors = {}
        
        detectors = [
            ('build_failures', self.detect_build_failures),
            ('test_failures', self.detect_test_failures),
            ('import_errors', self.detect_import_errors),
            ('echo_errors', self.detect_deep_tree_echo_errors)
        ]
        
        for error_type, detector_func in detectors:
            try:
                errors = detector_func()
                if errors:
                    all_errors[error_type] = errors
                    logger.info(f"Detected {len(errors)} {error_type}")
            except Exception as e:
                logger.error(f"Error in {error_type} detection: {e}")
                traceback.print_exc()
        
        return all_errors


class IssueCreator:
    """Creates GitHub issues for detected errors"""
    
    def __init__(self, github_token: str, repository: str):
        if not GITHUB_AVAILABLE:
            raise ImportError("PyGithub is required for issue creation")
        
        self.github = Github(github_token)
        self.repo = self.github.get_repo(repository)
        self.assignees = ['dtecho', 'drzo']
    
    def create_error_issue(self, error_data: Dict, error_category: str) -> Optional[int]:
        """Create a GitHub issue for an error"""
        try:
            title = self._generate_title(error_data, error_category)
            body = self._generate_body(error_data, error_category)
            labels = self._generate_labels(error_data, error_category)
            
            # Check for existing similar issues
            existing_issues = self.repo.get_issues(
                state='open',
                labels=['blocking-error']
            )
            
            for existing_issue in existing_issues:
                if self._is_similar_issue(existing_issue, error_data, error_category):
                    logger.info(f"Similar issue #{existing_issue.number} already exists")
                    self._update_existing_issue(existing_issue, error_data)
                    return existing_issue.number
            
            # Create new issue
            issue = self.repo.create_issue(
                title=title,
                body=body,
                labels=labels,
                assignees=self.assignees
            )
            
            logger.info(f"Created issue #{issue.number}: {title}")
            return issue.number
            
        except Exception as e:
            logger.error(f"Failed to create issue: {e}")
            traceback.print_exc()
            return None
    
    def _generate_title(self, error_data: Dict, error_category: str) -> str:
        """Generate issue title"""
        severity_emoji = {
            'low': '🟡',
            'medium': '🟠',
            'high': '🔴',
            'critical': '🚨'
        }
        
        severity = error_data.get('severity', 'medium')
        emoji = severity_emoji.get(severity, '⚠️')
        
        error_type = error_data.get('type', error_category)
        
        title_map = {
            'build_failure': 'Build System Failure',
            'test_failure': 'Test Suite Failure',
            'import_error': 'Module Import Failure',
            'dtesn_processor_error': 'DTESN Processor Error',
            'missing_echo_components': 'Missing Echo System Components',
            'cmake_configuration_error': 'CMake Configuration Error',
            'segmentation_fault': 'Critical Runtime Error (Segfault)',
            'module_error': 'Module Runtime Error'
        }
        
        title = title_map.get(error_type, 'System Error')
        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        
        return f"{emoji} [BLOCKING] {title} - {timestamp}"
    
    def _generate_body(self, error_data: Dict, error_category: str) -> str:
        """Generate detailed issue body"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        body = f"""## 🚨 Blocking Error Detected - Self-Healing System

**Auto-generated by the Self-Healing System**
**Detection Time**: {timestamp}
**Error Category**: {error_category}
**Error Type**: {error_data.get('type', 'unknown')}
**Severity**: {error_data.get('severity', 'medium').upper()}

### 📋 Error Details

"""
        
        # Add error-specific details
        if error_data.get('location'):
            body += f"**Location**: `{error_data['location']}`\n"
        
        if error_data.get('error'):
            body += f"**Error Message**: `{error_data['error']}`\n"
        
        if error_data.get('failed_tests'):
            body += "**Failed Tests**:\n"
            for test in error_data['failed_tests'][:5]:
                body += f"- `{test}`\n"
        
        if error_data.get('error_files'):
            body += "**Error Files**:\n"
            for file in error_data['error_files']:
                body += f"- `{file}`\n"
        
        if error_data.get('missing_directories'):
            body += "**Missing Components**:\n"
            for dir in error_data['missing_directories']:
                body += f"- `{dir}`\n"
        
        body += """
### 🛠️ Immediate Actions Required

#### For @dtecho and @drzo:
- [ ] **Acknowledge** this blocking error within 1 hour
- [ ] **Investigate** root cause using provided information
- [ ] **Assess** impact on Deep Tree Echo and Aphrodite systems
- [ ] **Implement** fix or workaround
- [ ] **Verify** resolution and system stability
- [ ] **Document** solution for future prevention

### 🔄 Suggested Recovery Steps

"""
        
        # Add error-specific recovery steps
        recovery_steps = self._get_recovery_steps(error_data.get('type', ''))
        body += recovery_steps
        
        body += f"""
### 📊 System Information
```json
{json.dumps(error_data, indent=2)}
```

### 🔗 Related Resources
- [Troubleshooting Guide](GITHUB_ACTIONS_GUIDE.md)
- [Development Setup](CONTRIBUTING.md)
- [Deep Tree Echo Architecture](DEEP_TREE_ECHO_ARCHITECTURE.md)

---
*🤖 Auto-generated by Self-Healing System | Priority: {error_data.get('severity', 'medium').upper()}*
"""
        
        return body
    
    def _get_recovery_steps(self, error_type: str) -> str:
        """Get recovery steps for specific error types"""
        steps = {
            'build_failure': """
1. **Clean Build Environment**:
   ```bash
   rm -rf build/ dist/ *.egg-info/
   ccache -C
   ```

2. **Check Dependencies**:
   ```bash
   pip install -r requirements/build.txt
   cmake --version  # Should be 3.26+
   ```

3. **Rebuild with Verbose Output**:
   ```bash
   export APHRODITE_TARGET_DEVICE=cpu
   pip install -e . --verbose
   ```
""",
            
            'test_failure': """
1. **Clean Test Environment**:
   ```bash
   find . -name "*.pyc" -delete
   find . -name "__pycache__" -exec rm -rf {} +
   rm -rf .pytest_cache/
   ```

2. **Run Specific Failed Tests**:
   ```bash
   pytest tests/ -v --tb=short -x
   ```

3. **Check Core Imports**:
   ```bash
   python -c "from aphrodite import LLM, SamplingParams; print('✅ Success')"
   ```
""",
            
            'import_error': """
1. **Check Python Environment**:
   ```bash
   python -c "import sys; print(sys.path)"
   pip list | grep aphrodite
   ```

2. **Reinstall Dependencies**:
   ```bash
   pip install -e . --force-reinstall
   ```

3. **Verify Installation**:
   ```bash
   python -c "import aphrodite; print(aphrodite.__version__)"
   ```
""",
            
            'dtesn_processor_error': """
1. **Check Echo System Status**:
   ```bash
   ls -la echo.*/
   python -c "from aphrodite.endpoints.deep_tree_echo import DTESNProcessor"
   ```

2. **Review DTESN Configuration**:
   ```bash
   grep -r "ERROR\\|FAILED" aphrodite/endpoints/deep_tree_echo/
   ```

3. **Reset Echo Systems**:
   ```bash
   export ECHO_ENABLE_DEEP_TREE=true
   export DEEP_TREE_ECHO_MODE=development
   ```
""",
            
            'missing_echo_components': """
1. **Verify Repository Integrity**:
   ```bash
   git status
   git submodule update --init --recursive
   ```

2. **Check Echo Directories**:
   ```bash
   ls -la echo.*/
   ```

3. **Restore Missing Components**:
   ```bash
   git checkout HEAD -- echo.*/
   ```
"""
        }
        
        return steps.get(error_type, """
1. **Check System Status**:
   ```bash
   ./quick-start.sh status
   ```

2. **Review Recent Changes**:
   ```bash
   git log --oneline -10
   ```

3. **Run General Diagnostics**:
   ```bash
   python --version
   pip check
   ```
""")
    
    def _generate_labels(self, error_data: Dict, error_category: str) -> List[str]:
        """Generate appropriate labels for the issue"""
        labels = [
            'blocking-error',
            'self-healing',
            f"severity-{error_data.get('severity', 'medium')}",
            f"category-{error_category}",
            'auto-generated'
        ]
        
        error_type = error_data.get('type', '')
        if 'build' in error_type:
            labels.append('build-system')
        if 'test' in error_type:
            labels.append('testing')
        if 'echo' in error_type or 'dtesn' in error_type:
            labels.append('deep-tree-echo')
        if 'import' in error_type:
            labels.append('python-environment')
        
        return labels
    
    def _is_similar_issue(self, existing_issue, error_data: Dict, error_category: str) -> bool:
        """Check if an existing issue is similar to the current error"""
        # Simple similarity check based on title keywords
        title_lower = existing_issue.title.lower()
        error_type = error_data.get('type', '').lower()
        
        similarity_keywords = [
            error_type,
            error_category.lower(),
            error_data.get('module', '').lower()
        ]
        
        return any(keyword and keyword in title_lower for keyword in similarity_keywords)
    
    def _update_existing_issue(self, issue, error_data: Dict):
        """Update existing issue with new error occurrence"""
        comment = f"""## 🔄 Additional Error Occurrence

**Detection Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

This error pattern has occurred again. Please prioritize resolution.

### Latest Error Data:
```json
{json.dumps(error_data, indent=2)}
```
"""
        issue.create_comment(comment)
        logger.info(f"Updated existing issue #{issue.number} with new occurrence")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Self-Healing System for Aphrodite Engine')
    parser.add_argument('--repo-path', default='.', help='Path to repository')
    parser.add_argument('--github-token', help='GitHub token for issue creation')
    parser.add_argument('--repository', help='GitHub repository (owner/repo)')
    parser.add_argument('--dry-run', action='store_true', help='Only detect errors, do not create issues')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize error detector
    detector = ErrorDetector(args.repo_path)
    
    # Detect all errors
    logger.info("Starting error detection...")
    all_errors = detector.detect_all_errors()
    
    if not all_errors:
        logger.info("✅ No blocking errors detected")
        return 0
    
    logger.info(f"🚨 Detected errors in {len(all_errors)} categories")
    
    # Print error summary
    for category, errors in all_errors.items():
        logger.info(f"  {category}: {len(errors)} errors")
        for error in errors:
            logger.info(f"    - {error.get('type', 'unknown')}: {error.get('severity', 'unknown')} severity")
    
    if args.dry_run:
        logger.info("Dry run mode - not creating issues")
        return 0
    
    # Create issues if GitHub credentials provided
    if args.github_token and args.repository:
        if not GITHUB_AVAILABLE:
            logger.error("PyGithub not available - cannot create issues")
            return 1
        
        logger.info("Creating GitHub issues...")
        issue_creator = IssueCreator(args.github_token, args.repository)
        
        created_issues = []
        for category, errors in all_errors.items():
            for error in errors:
                issue_number = issue_creator.create_error_issue(error, category)
                if issue_number:
                    created_issues.append(issue_number)
        
        if created_issues:
            logger.info(f"✅ Created/updated {len(created_issues)} issues: {created_issues}")
        else:
            logger.warning("No issues were created")
    else:
        logger.info("No GitHub credentials provided - skipping issue creation")
    
    return 1 if all_errors else 0


if __name__ == '__main__':
    sys.exit(main())