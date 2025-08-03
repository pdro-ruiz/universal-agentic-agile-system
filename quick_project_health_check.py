#!/usr/bin/env python3
"""
Quick Project Health Check - Optimized version for rapid analysis of large projects
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

class QuickHealthAnalyzer:
    """Quick project health analyzer"""
    
    def __init__(self):
        self.excluded_dirs = {
            '__pycache__', '.git', 'node_modules', 'venv', 'env', '.env',
            'dist', 'build', '.next', '.nuxt', 'target', 'bin', 'obj',
            'logs', 'log', 'tmp', 'temp', '.tmp', 'cache', '.cache',
            'backup', 'backups', 'archive', 'old', 'deprecated',
            '.vscode', '.idea', '.pytest_cache', 'coverage', '.coverage'
        }
        
        self.excluded_files = {
            '.pyc', '.pyo', '.class', '.o', '.so', '.dll', '.exe',
            '.log', '.out', '.err', '.db', '.sqlite', '.sqlite3'
        }
        
        self.deprecated_patterns = [
            (r'.*_old\b', 'Old version file'),
            (r'.*_deprecated\b', 'Deprecated file'),
            (r'.*_legacy\b', 'Legacy file'),
            (r'.*\.backup\b', 'Backup file'),
            (r'.*\.bak\b', 'Backup file'),
            (r'.*copy.*', 'Possible duplicate'),
            (r'.*\s\(\d+\)', 'Numbered duplicate')
        ]
    
    def should_exclude(self, path: Path) -> bool:
        """Checks if the item should be excluded"""
        
        # Exclude technical directories
        if path.is_dir() and path.name.lower() in self.excluded_dirs:
            return True
        
        # Exclude files by extension
        if path.is_file() and path.suffix.lower() in self.excluded_files:
            return True
        
        # Exclude very small files (likely not important)
        if path.is_file():
            try:
                if path.stat().st_size < 10:  # Files smaller than 10 bytes
                    return True
            except OSError:
                pass
        
        return False
    
    def analyze_project_quick(self, project_path: str) -> Dict[str, Any]:
        """Quick project analysis"""
        
        project_root = Path(project_path)
        if not project_root.exists():
            return {"error": f"Path does not exist: {project_path}"}
        
        print(f"QUICK ANALYSIS: {project_root.name}")
        print("="*50)
        
        stats = {
            "total_scanned": 0,
            "valid_elements": 0,
            "technical_filtered": 0,
            "deprecated_found": 0,
            "suspicious_files": [],
            "very_old_files": [],
            "large_files": [],
            "duplicate_candidates": [],
            "health_issues": []
        }
        
        # Limit to avoid very long analysis
        max_files = 1000
        two_years_ago = datetime.now() - timedelta(days=730)
        
        try:
            for item in project_root.rglob("*"):
                stats["total_scanned"] += 1
                
                # Safety limit
                if stats["total_scanned"] > max_files:
                    stats["health_issues"].append(f"Large project: stopped analysis at {max_files} files")
                    break
                
                # Filter technical elements
                if self.should_exclude(item):
                    stats["technical_filtered"] += 1
                    continue
                
                stats["valid_elements"] += 1
                
                # Check deprecated patterns
                item_str = str(item).lower()
                for pattern, description in self.deprecated_patterns:
                    if re.search(pattern, item_str):
                        stats["deprecated_found"] += 1
                        stats["suspicious_files"].append({
                            "path": str(item),
                            "name": item.name,
                            "reason": description,
                            "type": "deprecated_pattern"
                        })
                        break
                
                # Check very old files (files only)
                if item.is_file():
                    try:
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        size = item.stat().st_size
                        
                        if mtime < two_years_ago:
                            stats["very_old_files"].append({
                                "path": str(item),
                                "name": item.name,
                                "last_modified": mtime.strftime('%Y-%m-%d'),
                                "age_days": (datetime.now() - mtime).days
                            })
                        
                        # Very large files (>1MB)
                        if size > 1024 * 1024:
                            stats["large_files"].append({
                                "path": str(item),
                                "name": item.name,
                                "size_mb": round(size / (1024 * 1024), 2)
                            })
                        
                    except OSError:
                        pass
        
        except Exception as e:
            stats["health_issues"].append(f"Analysis error: {str(e)}")
        
        # Detect possible duplicates by name
        file_names = {}
        for item in stats["suspicious_files"]:
            base_name = re.sub(r'\s?\(\d+\)|\s?copy|\s?-\s?copy', '', item["name"].lower())
            if base_name in file_names:
                if base_name not in [d["base_name"] for d in stats["duplicate_candidates"]]:
                    stats["duplicate_candidates"].append({
                        "base_name": base_name,
                        "files": [file_names[base_name], item["name"]],
                        "count": 2
                    })
                else:
                    # Add to existing duplicate
                    for dup in stats["duplicate_candidates"]:
                        if dup["base_name"] == base_name:
                            dup["files"].append(item["name"])
                            dup["count"] += 1
                            break
            else:
                file_names[base_name] = item["name"]
        
        # Calculate health metrics
        if stats["valid_elements"] > 0:
            deprecated_ratio = stats["deprecated_found"] / stats["valid_elements"]
            old_files_ratio = len(stats["very_old_files"]) / stats["valid_elements"]
            
            # Health score simple
            health_score = max(0.0, 1.0 - deprecated_ratio - old_files_ratio * 0.5)
        else:
            health_score = 1.0
        
        # Generate recommendations
        recommendations = []
        if stats["deprecated_found"] > 0:
            recommendations.append(f"Review {stats['deprecated_found']} files with deprecated patterns")
        
        if len(stats["very_old_files"]) > 10:
            recommendations.append(f"Evaluate {len(stats['very_old_files'])} very old files (>2 years)")
        
        if len(stats["large_files"]) > 5:
            recommendations.append(f"Review {len(stats['large_files'])} large files (>1MB)")
        
        if len(stats["duplicate_candidates"]) > 0:
            recommendations.append(f"Investigate {len(stats['duplicate_candidates'])} possible duplicates")
        
        # Final result
        result = {
            "project_name": project_root.name,
            "analysis_summary": {
                "total_scanned": stats["total_scanned"],
                "valid_elements": stats["valid_elements"],
                "technical_filtered": stats["technical_filtered"],
                "health_score": round(health_score * 100, 1),
                "analysis_completed": stats["total_scanned"] < max_files
            },
            "issues_found": {
                "deprecated_patterns": stats["deprecated_found"],
                "very_old_files": len(stats["very_old_files"]),
                "large_files": len(stats["large_files"]),
                "possible_duplicates": len(stats["duplicate_candidates"])
            },
            "recommendations": recommendations,
            "sample_issues": {
                "deprecated_files": stats["suspicious_files"][:5],
                "old_files": stats["very_old_files"][:5],
                "large_files": stats["large_files"][:5],
                "duplicate_candidates": stats["duplicate_candidates"][:3]
            }
        }
        
        # Print results
        print(f"Analysis completed:")
        print(f"   Valid elements: {result['analysis_summary']['valid_elements']}")
        print(f"   Filtered: {result['analysis_summary']['technical_filtered']}")
        print(f"   Health Score: {result['analysis_summary']['health_score']}%")
        print(f"   Issues found: {sum(result['issues_found'].values())}")
        
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        return result

def quick_demo(project_path: str):
    """Quick analyzer demo"""
    
    analyzer = QuickHealthAnalyzer()
    result = analyzer.analyze_project_quick(project_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\nDETAILS FOUND:")
    
    # Show deprecated files
    if result["sample_issues"]["deprecated_files"]:
        print(f"\nFiles with suspicious patterns:")
        for file in result["sample_issues"]["deprecated_files"]:
            print(f"   - {file['name']}: {file['reason']}")
    
    # Show old files
    if result["sample_issues"]["old_files"]:
        print(f"\nVery old files:")
        for file in result["sample_issues"]["old_files"]:
            print(f"   - {file['name']}: {file['last_modified']} ({file['age_days']} days)")
    
    # Show large files
    if result["sample_issues"]["large_files"]:
        print(f"\nLarge files:")
        for file in result["sample_issues"]["large_files"]:
            print(f"   - {file['name']}: {file['size_mb']} MB")
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
        quick_demo(project_path)
    else:
        print("Usage: python quick_project_health_check.py /path/to/project")
        print("Example: python quick_project_health_check.py /mnt/c/Users/pedro/Desktop/agente/Diamond-Team-Clean")