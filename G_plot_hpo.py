import os
import optuna
import optuna.visualization as vis
import config

def generate_thesis_plots():
    db_path = os.path.join(config.RESULTS_DIR, "optuna_study.db")
    
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at {db_path}")
        print("Please run HPO_optuna.py first to generate some trials.")
        return

    print(f"Loading study from {db_path}...")
    
    try:
        # Load the saved study from the database
        study = optuna.load_study(
            study_name="vgg_swin_hpo_v2", 
            storage=f"sqlite:///{db_path}"
        )
        
        print(f"Study loaded! Total trials found: {len(study.trials)}")
        
        if len(study.trials) == 0:
            print("The study is empty. Run HPO_optuna.py to train some trials first.")
            return

        print("\nGenerating interactive HTML plots...")
        
        # 1. Optimization History
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(f"{config.RESULTS_DIR}/hpo_optimization_history.html")
        print("  ✅ Optimization History saved.")
        
        # 2. Hyperparameter Importances
        fig_importances = vis.plot_param_importances(study)
        fig_importances.write_html(f"{config.RESULTS_DIR}/hpo_param_importances.html")
        print("  ✅ Parameter Importances saved.")
        
        # 3. Parallel Coordinate Plot
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_parallel.write_html(f"{config.RESULTS_DIR}/hpo_parallel_coordinate.html")
        print("  ✅ Parallel Coordinate Plot saved.")
        
        # 4. Slice Plot
        fig_slice = vis.plot_slice(study)
        fig_slice.write_html(f"{config.RESULTS_DIR}/hpo_slice.html")
        print("  ✅ Slice Plot saved.")
        
        print(f"\n🎉 All interactive visualization plots have been saved to the '{config.RESULTS_DIR}/' folder!")
        print("You can open these HTML files in any web browser to view, interact with, and screenshot them for your thesis.")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Make sure you have plotly installed: pip install plotly")

if __name__ == "__main__":
    generate_thesis_plots()
