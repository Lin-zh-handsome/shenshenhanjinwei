"""Regenerate paper indexes and small tables from existing repository artifacts."""
import build_artifact_indexes
import build_experiment_registry
import build_final_results
import build_q2_tables
import build_q3_tables
import build_repository_inventory
import check_result_consistency


def main():
    build_final_results.main()
    build_q2_tables.main()
    build_q3_tables.main()
    build_experiment_registry.main()
    build_artifact_indexes.main()
    build_repository_inventory.main()
    check_result_consistency.main()


if __name__ == "__main__":
    main()
