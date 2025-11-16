"""
Master script to run the complete requirement extraction pipeline
"""

import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

def run_step(script_name: str, description: str, args: list = None):
    """Run a pipeline step"""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    
    cmd = [sys.executable, str(SCRIPT_DIR / script_name)]
    if args:
        cmd.extend(args)
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Error in {description}")
        print(f"Pipeline stopped.")
        sys.exit(1)
    
    print(f"\n✅ {description} completed successfully")


def main():
    """Run the complete pipeline"""
    print("\n" + "=" * 70)
    print("RISK REQUIREMENT EXTRACTION - FULL PIPELINE")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Extract requirements from all documents (sliding window)")
    print("  2. Find overlaps and contradictions between requirements")
    print("  3. Export results for frontend visualization")
    print("\n⚠️  This may take a significant amount of time!")
    print("=" * 70)
    
    # Check if user wants to limit documents or force reprocess
    limit = None
    force_reprocess = False
    skip_extraction = True

    for arg in sys.argv[1:]:
        if arg == "--force" or arg == "--reprocess":
            force_reprocess = True
        elif arg == "--skip-extraction":
            skip_extraction = True
        elif arg.isdigit():
            limit = int(arg)

    if limit:
        print(f"\n📝 Note: Processing only {limit} documents (test mode)")
    if force_reprocess:
        print("  🔄 Force reprocessing enabled - will reprocess all documents")
    if skip_extraction:
        print("  ⏭️  Skipping requirement extraction - using existing requirements")
    
    # Step 1: Extract requirements (optional)
    if not skip_extraction:
        args = []
        if limit:
            args.append(str(limit))
        if force_reprocess:
            args.append("--force")
        run_step(
            "requirement_extraction_pipeline.py",
            "Extract Requirements from Documents",
            args
        )
    else:
        print("\n⏭️  Skipping requirement extraction step")
    
    # Step 2: Find relationships
    run_step(
        "requirement_relationship_pipeline.py",
        "Find Overlaps and Contradictions"
    )
    
    # Step 3: Export for frontend
    run_step(
        "export_for_frontend.py",
        "Export Results for Frontend"
    )
    
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Start the backend API: python backend_api.py")
    print("  2. Open http://localhost:5000 in your browser")
    print("  3. View requirements-based analysis")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

