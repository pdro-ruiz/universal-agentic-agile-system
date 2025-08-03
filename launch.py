#!/usr/bin/env python3
"""
Universal Agile System - Primary System Launcher
Single entry point for the universal agile system
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Import core components only
from universal_project_detector import UniversalProjectDetector
from quick_project_health_check import QuickHealthAnalyzer  
from deep_agile_methodology import demo_deep_agile
from universal_agile_system_final import demo_universal_agile_final

def print_banner():
    """Display system banner"""
    print("""
UNIVERSAL AGILE SYSTEM - PRODUCTION READY
═══════════════════════════════════════════════
    Automatic Detection + Deep Agile Methodology
    Intelligent Autosave + Context Liberation
    Health Analysis + Advanced Filters
═══════════════════════════════════════════════
""")

async def quick_health_analysis(project_path: str):
    """Quick health analysis (30 seconds)"""
    print("RAPID HEALTH ANALYSIS")
    print("="*40)
    
    analyzer = QuickHealthAnalyzer()
    result = analyzer.analyze_project_quick(project_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Project analyzed: {result['project_name']}")
    print(f"Health Score: {result['analysis_summary']['health_score']}%")
    
    if result["recommendations"]:
        print(f"\nRecommendations:")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    return result

async def detect_structure(project_path: str):
    """Structure detection (1 minute)"""
    print("STRUCTURE DETECTION")
    print("="*40)
    
    detector = UniversalProjectDetector()
    detected = await detector.detect_project_structure(project_path)
    
    print(f"Project: {detected.name}")
    print(f"Type: {detected.project_type.value}")
    print(f"Structure: {detected.structure_depth.value} levels")
    print(f"Agents/members: {detected.total_agents}")
    print(f"Squads/teams: {len(detected.squads)}")
    print(f"Coordinators: {len(detected.coordinators)}")
    print(f"Confidence: {detected.detection_confidence*100:.1f}%")
    
    return detected

async def demo_methodology():
    """Agile methodology demo (2 minutes)"""
    print("DEMO: DEEP AGILE METHODOLOGY")
    print("="*40)
    
    orchestrator = await demo_deep_agile()
    
    print(f"\nDEMO COMPLETED:")
    print(f"   Epics created: {len(orchestrator.epics)}")
    print(f"   User Stories: {len(orchestrator.product_backlog.user_stories)}")
    print(f"   Grooming sessions: {len(orchestrator.grooming_sessions)}")
    print(f"   Planning poker: {len(orchestrator.planning_poker_sessions)}")
    
    return orchestrator

async def full_system_demo():
    """Complete system demo (5 minutes)"""
    print("COMPLETE INTEGRATED SYSTEM")
    print("="*40)
    
    system, dashboard = await demo_universal_agile_final()
    
    print(f"\nSYSTEM DEPLOYED:")
    print(f"   Projects: {dashboard['system_health']['total_active_projects']}")
    print(f"   Average health: {dashboard['global_metrics']['average_health_score']:.1f}%")
    print(f"   Total capacity: {dashboard['system_health']['total_capacity']}")
    print(f"   Epics: {dashboard['global_metrics']['total_epics']}")
    print(f"   User Stories: {dashboard['global_metrics']['total_user_stories']}")
    
    return system

def interactive_menu():
    """Main interactive menu"""
    
    while True:
        print_banner()
        print("SELECT AN OPTION:")
        print("="*30)
        print("1. Rapid health analysis (30s)")
        print("2. Structure detection (1m)")  
        print("3. Agile methodology demo (2m)")
        print("4. Complete integrated system (5m)")
        print("5. Exit")
        
        choice = input("\nOption (1-5): ").strip()
        
        if choice == "1":
            project_path = input("Project path: ").strip()
            if project_path:
                asyncio.run(quick_health_analysis(project_path))
                input("\nPress Enter to continue...")
        
        elif choice == "2":
            project_path = input("Project path: ").strip()
            if project_path:
                asyncio.run(detect_structure(project_path))
                input("\nPress Enter to continue...")
        
        elif choice == "3":
            asyncio.run(demo_methodology())
            input("\nPress Enter to continue...")
        
        elif choice == "4":
            asyncio.run(full_system_demo())
            input("\nPress Enter to continue...")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option")
            input("Press Enter to continue...")

async def main():
    """Main function with command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Universal Agile System - Production Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python3 launch.py                                    # Interactive menu
  python3 launch.py --health /path/to/project          # Quick analysis
  python3 launch.py --detect /path/to/project          # Structure detection
  python3 launch.py --demo-agile                       # Methodology demo
  python3 launch.py --full-demo                        # Complete system
        """
    )
    
    parser.add_argument("--health", metavar="PATH", 
                       help="Quick project health analysis")
    parser.add_argument("--detect", metavar="PATH",
                       help="Automatic structure detection")
    parser.add_argument("--demo-agile", action="store_true",
                       help="Deep agile methodology demo")
    parser.add_argument("--full-demo", action="store_true",
                       help="Complete integrated system demo")
    
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.health:
        await quick_health_analysis(args.health)
    elif args.detect:
        await detect_structure(args.detect)
    elif args.demo_agile:
        await demo_methodology()
    elif args.full_demo:
        await full_system_demo()
    else:
        # Default interactive menu
        interactive_menu()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)