#!/bin/bash

# BTC News Recommendation Experiments Runner
# This script runs experiments on MIND-small dataset sequentially

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default experiments (can be overridden by command line arguments)
DEFAULT_EXPERIMENTS=(
    "naml_mind_small"
    "nrms_mind_small"
)

# Function to print colored output
print_status() {
    echo -e "${1}${2}${NC}"
}

# Function to print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS] [EXPERIMENTS...]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -l, --list          List available experiments"
    echo "  -d, --delay SECONDS Set delay between experiments (default: 5)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run default experiments"
    echo "  $0 naml_mind_small                    # Run single experiment"
    echo "  $0 naml_mind_small nrms_mind_small    # Run specific experiments"
    echo "  $0 -d 10 naml_mind_small nrms_mind_small  # Run with 10s delay"
    echo ""
    echo "Available experiments:"
    echo "  - naml_mind_small"
    echo "  - nrms_mind_small"
    echo "  - (add more as they become available)"
}

# Function to list available experiments
list_experiments() {
    print_status "$CYAN" "Available experiments:"
    echo ""
    
    # Check for experiment config files
    if [ -d "configs/experiment" ]; then
        for config_file in configs/experiment/*.yaml; do
            if [ -f "$config_file" ]; then
                experiment_name=$(basename "$config_file" .yaml)
                print_status "$GREEN" "  ✓ $experiment_name"
            fi
        done
    else
        print_status "$RED" "  No experiment configs found in configs/experiment/"
    fi
    
    echo ""
    print_status "$YELLOW" "Default experiments:"
    for exp in "${DEFAULT_EXPERIMENTS[@]}"; do
        echo "  - $exp"
    done
}

# Function to validate experiment exists
validate_experiment() {
    local experiment_name=$1
    
    if [ ! -f "configs/experiment/${experiment_name}.yaml" ]; then
        print_status "$RED" "Error: Experiment '$experiment_name' not found."
        print_status "$RED" "Config file 'configs/experiment/${experiment_name}.yaml' does not exist."
        return 1
    fi
    return 0
}

# Function to run a single experiment
run_experiment() {
    local experiment_name=$1
    
    print_status "$BLUE" "Starting experiment: $experiment_name"
    print_status "$CYAN" "Configuration: configs/experiment/${experiment_name}.yaml"
    
    local start_time=$(date +%s)
    
    # Run the experiment
    if python src/train.py experiment="$experiment_name"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        print_status "$GREEN" "✓ Experiment $experiment_name completed successfully!"
        print_status "$GREEN" "Duration: ${duration}s (${minutes}m ${seconds}s)"
        return 0
    else
        print_status "$RED" "✗ Experiment $experiment_name failed!"
        return 1
    fi
}

# Main execution
main() {
    local delay_seconds=5
    local experiments=()
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -l|--list)
                list_experiments
                exit 0
                ;;
            -d|--delay)
                delay_seconds="$2"
                shift 2
                ;;
            -*)
                print_status "$RED" "Unknown option: $1"
                print_usage
                exit 1
                ;;
            *)
                experiments+=("$1")
                shift
                ;;
        esac
    done
    
    # Use default experiments if none specified
    if [ ${#experiments[@]} -eq 0 ]; then
        experiments=("${DEFAULT_EXPERIMENTS[@]}")
        print_status "$YELLOW" "No experiments specified, using defaults:"
        for exp in "${experiments[@]}"; do
            echo "  - $exp"
        done
        echo ""
    fi
    
    print_status "$YELLOW" "=== BTC News Recommendation Experiments ==="
    echo "This script will run ${#experiments[@]} experiment(s) on MIND-small dataset"
    print_status "$CYAN" "Delay between experiments: ${delay_seconds}s"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "src/train.py" ]; then
        print_status "$RED" "Error: src/train.py not found. Please run this script from the project root directory."
        exit 1
    fi
    
    # Validate all experiments before starting
    print_status "$BLUE" "Validating experiments..."
    for experiment in "${experiments[@]}"; do
        if ! validate_experiment "$experiment"; then
            exit 1
        fi
    done
    print_status "$GREEN" "✓ All experiments validated successfully!"
    echo ""
    
    # Track results
    total_start_time=$(date +%s)
    failed_experiments=()
    
    # Run experiments
    for i in "${!experiments[@]}"; do
        experiment_name="${experiments[$i]}"
        experiment_num=$((i + 1))
        total_experiments=${#experiments[@]}
        
        echo ""
        print_status "$CYAN" "Experiment ${experiment_num}/${total_experiments}: $experiment_name"
        
        if run_experiment "$experiment_name"; then
            print_status "$GREEN" "✓ $experiment_name experiment passed"
        else
            print_status "$RED" "✗ $experiment_name experiment failed"
            failed_experiments+=("$experiment_name")
        fi
        
        # Add delay between experiments (except for the last one)
        if [ $i -lt $((${#experiments[@]} - 1)) ]; then
            print_status "$YELLOW" "Waiting ${delay_seconds} seconds before next experiment..."
            sleep "$delay_seconds"
        fi
    done
    
    # Summary
    total_end_time=$(date +%s)
    total_duration=$((total_end_time - total_start_time))
    total_minutes=$((total_duration / 60))
    total_seconds=$((total_duration % 60))
    
    echo ""
    print_status "$YELLOW" "=== Experiment Summary ==="
    print_status "$CYAN" "Total duration: ${total_duration}s (${total_minutes}m ${total_seconds}s)"
    echo ""
    
    # Report results
    if [ ${#failed_experiments[@]} -eq 0 ]; then
        print_status "$GREEN" "🎉 All experiments completed successfully!"
        for experiment in "${experiments[@]}"; do
            print_status "$GREEN" "✓ $experiment: PASSED"
        done
        exit 0
    else
        print_status "$RED" "❌ Some experiments failed:"
        for failed in "${failed_experiments[@]}"; do
            print_status "$RED" "✗ $failed: FAILED"
        done
        exit 1
    fi
}

# Handle interrupts gracefully
trap 'echo -e "\n${YELLOW}⚠️  Experiments interrupted by user${NC}"; exit 1' INT TERM

# Run main function
main "$@" 