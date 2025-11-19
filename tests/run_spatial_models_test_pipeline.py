#!/usr/bin/env python3
"""
Automated test pipeline for spatial prediction models.
Runs comprehensive testing including unit tests, integration tests, performance benchmarks, and coverage analysis.
"""

import os
import sys
import subprocess
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/results/test_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestPipeline:
    """Automated test pipeline for spatial models"""
    
    def __init__(self, test_dir: str = "tests", results_dir: str = "tests/results",
                 coverage_threshold: float = 95.0, parallel: bool = True):
        self.test_dir = Path(test_dir)
        self.results_dir = Path(results_dir)
        self.coverage_threshold = coverage_threshold
        self.parallel = parallel
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Test categories
        self.test_categories = {
            'unit': {
                'description': 'Unit tests for individual models',
                'pattern': 'test_spatial_models.py',
                'coverage_target': 'models',
                'required': True
            },
            'integration': {
                'description': 'Integration tests with data pipelines',
                'pattern': 'test_spatial_models_integration.py',
                'coverage_target': 'models',
                'required': True
            },
            'benchmark': {
                'description': 'Performance benchmarks',
                'pattern': 'test_spatial_models_performance.py',
                'coverage_target': None,
                'required': False
            },
            'e2e': {
                'description': 'End-to-end training pipeline tests',
                'pattern': 'test_spatial_models_integration.py::TestEndToEndPipeline',
                'coverage_target': None,
                'required': True
            }
        }
        
        self.results = {
            'start_time': datetime.now().isoformat(),
            'test_categories': {},
            'overall_status': 'unknown',
            'coverage_summary': {},
            'performance_summary': {},
            'recommendations': []
        }
    
    def run_command(self, cmd: List[str], timeout: int = 1800) -> Tuple[bool, str, str]:
        """Run a command and return success status, stdout, and stderr"""
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.test_dir.parent
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            if not success:
                logger.warning(f"Command failed with return code {result.returncode}")
                if stderr:
                    logger.error(f"Error output: {stderr[:500]}")
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds")
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            return False, "", str(e)
    
    def run_unit_tests(self) -> Dict[str, any]:
        """Run unit tests with coverage"""
        logger.info("Running unit tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            str(self.test_dir / 'unit' / 'test_spatial_models.py'),
            '-v', '--tb=short',
            f'--cov=models',
            f'--cov-report=html:{self.results_dir}/coverage_html',
            f'--cov-report=term-missing',
            f'--cov-report=json:{self.results_dir}/coverage.json',
            f'--cov-fail-under={self.coverage_threshold}'
        ]
        
        if self.parallel:
            cmd.extend(['-n', 'auto'])
        
        success, stdout, stderr = self.run_command(cmd)
        
        # Parse coverage results
        coverage_data = self.parse_coverage_results(f'{self.results_dir}/coverage.json')
        
        return {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'coverage': coverage_data,
            'test_type': 'unit'
        }
    
    def run_integration_tests(self) -> Dict[str, any]:
        """Run integration tests"""
        logger.info("Running integration tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            str(self.test_dir / 'integration' / 'test_spatial_models_integration.py'),
            '-v', '--tb=short',
            '-m', 'integration',
            '--durations=10'
        ]
        
        if self.parallel:
            cmd.extend(['-n', '1'])  # Integration tests should run sequentially
        
        success, stdout, stderr = self.run_command(cmd)
        
        return {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'test_type': 'integration'
        }
    
    def run_benchmark_tests(self) -> Dict[str, any]:
        """Run performance benchmarks"""
        logger.info("Running performance benchmarks...")
        
        cmd = [
            'python', '-m', 'pytest',
            str(self.test_dir / 'benchmark' / 'test_spatial_models_performance.py'),
            '-v', '--tb=short',
            '-m', 'benchmark',
            '--durations=0'
        ]
        
        success, stdout, stderr = self.run_command(cmd, timeout=3600)  # 1 hour timeout
        
        # Parse benchmark results if available
        benchmark_file = self.results_dir / 'full_spatial_models_benchmark.json'
        benchmark_data = {}
        if benchmark_file.exists():
            try:
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse benchmark results: {e}")
        
        return {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'benchmark_data': benchmark_data,
            'test_type': 'benchmark'
        }
    
    def run_e2e_tests(self) -> Dict[str, any]:
        """Run end-to-end tests"""
        logger.info("Running end-to-end tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            str(self.test_dir / 'integration' / 'test_spatial_models_integration.py'),
            '-v', '--tb=short',
            '-m', 'e2e',
            '--durations=10'
        ]
        
        success, stdout, stderr = self.run_command(cmd, timeout=1800)  # 30 minute timeout
        
        return {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'test_type': 'e2e'
        }
    
    def parse_coverage_results(self, coverage_file: str) -> Dict[str, any]:
        """Parse coverage results from JSON file"""
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            # Extract relevant information
            total_statements = coverage_data.get('totals', {}).get('statements', 0)
            covered_statements = coverage_data.get('totals', {}).get('covered_statements', 0)
            coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)
            
            # Get file-level coverage
            file_coverage = {}
            for file_path, file_data in coverage_data.get('files', {}).items():
                if 'models' in file_path:
                    file_coverage[file_path] = {
                        'coverage_percent': file_data.get('summary', {}).get('percent_covered', 0),
                        'statements': file_data.get('summary', {}).get('num_statements', 0),
                        'covered': file_data.get('summary', {}).get('covered_statements', 0)
                    }
            
            return {
                'total_statements': total_statements,
                'covered_statements': covered_statements,
                'coverage_percent': coverage_percent,
                'file_coverage': file_coverage,
                'meets_threshold': coverage_percent >= self.coverage_threshold
            }
            
        except Exception as e:
            logger.error(f"Error parsing coverage results: {e}")
            return {
                'total_statements': 0,
                'covered_statements': 0,
                'coverage_percent': 0,
                'file_coverage': {},
                'meets_threshold': False,
                'error': str(e)
            }
    
    def generate_recommendations(self, results: Dict[str, any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Coverage recommendations
        if 'coverage' in results.get('unit', {}):
            coverage = results['unit']['coverage']
            if not coverage.get('meets_threshold', False):
                current_coverage = coverage.get('coverage_percent', 0)
                recommendations.append(
                    f"Code coverage is {current_coverage:.1f}%, below threshold of {self.coverage_threshold}%. "
                    "Consider adding more unit tests for uncovered code."
                )
            
            # Check for low-coverage files
            low_coverage_files = []
            for file_path, file_cov in coverage.get('file_coverage', {}).items():
                if file_cov['coverage_percent'] < 80.0:  # 80% is our target for individual files
                    low_coverage_files.append((file_path, file_cov['coverage_percent']))
            
            if low_coverage_files:
                recommendations.append(
                    f"Found {len(low_coverage_files)} files with <80% coverage. "
                    "Focus on improving coverage for these files."
                )
        
        # Performance recommendations
        if 'benchmark' in results and results['benchmark']['success']:
            benchmark_data = results['benchmark'].get('benchmark_data', [])
            if benchmark_data:
                # Analyze performance data and make recommendations
                slow_models = [item for item in benchmark_data if item.get('inference_time_ms', 0) > 100]
                if slow_models:
                    recommendations.append(
                        f"Found {len(slow_models)} models with inference time >100ms. "
                        "Consider optimizing these models for production use."
                    )
        
        # Integration test recommendations
        if 'integration' in results and not results['integration']['success']:
            recommendations.append(
                "Integration tests failed. Check model compatibility with data pipelines "
                "and ensure proper error handling."
            )
        
        # E2E test recommendations
        if 'e2e' in results and not results['e2e']['success']:
            recommendations.append(
                "End-to-end tests failed. Verify complete training pipeline "
                "including data loading, model training, and evaluation."
            )
        
        return recommendations
    
    def save_results(self, results: Dict[str, any]):
        """Save test results to JSON file"""
        results_file = self.results_dir / 'test_pipeline_results.json'
        
        # Add completion timestamp
        results['end_time'] = datetime.now().isoformat()
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def generate_html_report(self, results: Dict[str, any]):
        """Generate HTML test report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Spatial Models Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
        .failure {{ background-color: #f8d7da; border-color: #f5c6cb; }}
        .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
        .recommendation {{ margin: 10px 0; padding: 10px; background-color: #e2e3e5; border-radius: 3px; }}
        pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Spatial Models Test Report</h1>
        <p><strong>Generated:</strong> {results['start_time']}</p>
        <p><strong>Overall Status:</strong> {results['overall_status']}</p>
    </div>
"""
        
        # Add coverage section
        if 'unit' in results and 'coverage' in results['unit']:
            coverage = results['unit']['coverage']
            html_content += f"""
    <div class="section {'success' if coverage['meets_threshold'] else 'failure'}">
        <h2>Code Coverage</h2>
        <div class="metric">Total Statements: {coverage['total_statements']}</div>
        <div class="metric">Covered Statements: {coverage['covered_statements']}</div>
        <div class="metric">Coverage Percentage: {coverage['coverage_percent']:.1f}%</div>
        <div class="metric">Meets Threshold: {coverage['meets_threshold']}</div>
    </div>
"""
        
        # Add test results sections
        for test_type, test_result in results['test_categories'].items():
            status_class = 'success' if test_result['success'] else 'failure'
            html_content += f"""
    <div class="section {status_class}">
        <h2>{test_type.title()} Tests</h2>
        <p><strong>Status:</strong> {'Passed' if test_result['success'] else 'Failed'}</p>
        <p><strong>Duration:</strong> {test_result.get('duration', 'Unknown')}</p>
        {f'<pre>{test_result.get("stdout", "")[:1000]}</pre>' if test_result.get('stdout') else ''}
    </div>
"""
        
        # Add recommendations
        if results['recommendations']:
            html_content += """
    <div class="section">
        <h2>Recommendations</h2>
"""
            for rec in results['recommendations']:
                html_content += f"        <div class=\"recommendation\">{rec}</div>\n"
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML report
        html_file = self.results_dir / 'test_report.html'
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_file}")
    
    def run_pipeline(self, categories: List[str] = None) -> Dict[str, any]:
        """Run the complete test pipeline"""
        logger.info("Starting spatial models test pipeline...")
        
        # Use all categories if none specified
        if categories is None:
            categories = list(self.test_categories.keys())
        
        # Validate categories
        invalid_categories = [cat for cat in categories if cat not in self.test_categories]
        if invalid_categories:
            logger.error(f"Invalid test categories: {invalid_categories}")
            return {'error': f'Invalid categories: {invalid_categories}'}
        
        overall_success = True
        
        for category in categories:
            logger.info(f"Running {category} tests...")
            
            start_time = time.time()
            
            if category == 'unit':
                result = self.run_unit_tests()
            elif category == 'integration':
                result = self.run_integration_tests()
            elif category == 'benchmark':
                result = self.run_benchmark_tests()
            elif category == 'e2e':
                result = self.run_e2e_tests()
            else:
                logger.warning(f"Unknown test category: {category}")
                continue
            
            duration = time.time() - start_time
            result['duration'] = f"{duration:.2f}s"
            
            self.results['test_categories'][category] = result
            
            # Check if this category is required and failed
            if self.test_categories[category]['required'] and not result['success']:
                overall_success = False
                logger.error(f"Required test category '{category}' failed")
        
        # Generate recommendations
        self.results['recommendations'] = self.generate_recommendations(self.results['test_categories'])
        
        # Set overall status
        self.results['overall_status'] = 'success' if overall_success else 'failure'
        
        # Save results
        self.save_results(self.results)
        
        # Generate HTML report
        self.generate_html_report(self.results)
        
        logger.info(f"Test pipeline completed. Overall status: {self.results['overall_status']}")
        
        return self.results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Spatial Models Test Pipeline")
    parser.add_argument('--categories', nargs='+', 
                       choices=['unit', 'integration', 'benchmark', 'e2e'],
                       help='Test categories to run (default: all)')
    parser.add_argument('--coverage-threshold', type=float, default=95.0,
                       help='Minimum coverage threshold (default: 95.0)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel test execution')
    parser.add_argument('--results-dir', type=str, default='tests/results',
                       help='Directory for test results (default: tests/results)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create test pipeline
    pipeline = TestPipeline(
        coverage_threshold=args.coverage_threshold,
        parallel=not args.no_parallel,
        results_dir=args.results_dir
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(args.categories)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST PIPELINE SUMMARY")
    print("="*80)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Results saved to: {args.results_dir}")
    
    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"- {rec}")
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_status'] == 'success' else 1)


if __name__ == "__main__":
    main()