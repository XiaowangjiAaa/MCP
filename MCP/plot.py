import os
import pandas as pd
import matplotlib.pyplot as plt
#from mcp.server.fastmcp import tool
from MCP.tool import tool

# placeholder decorator
def tool(name=None):
    def decorator(fn):
        return fn
    return decorator


@tool(name="plot_comparison_graphs")
def plot_comparison_graphs(gt_csv_path: str, pred_csv_path: str, output_dir: str = "outputs/figures") -> dict:
    """Plot prediction vs GT scatter charts and save the figures."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        df_gt = pd.read_csv(gt_csv_path).set_index("Image")
        df_pred = pd.read_csv(pred_csv_path).set_index("Image")
        df = df_gt.join(df_pred, lsuffix="_gt", rsuffix="_pred").dropna()

        if df.empty:
            raise ValueError("No comparable images; GT and prediction mismatch")

        metrics = [
            ("Length (mm)", "green", "length_comparison.png"),
            ("Area (mm^2)", "blue", "area_comparison.png"),
            ("Max Width (mm)", "red", "max_width_comparison.png"),
            ("Avg Width (mm)", "magenta", "avg_width_comparison.png"),
        ]

        combined_path = os.path.join(output_dir, "comparison_all_2x2.png")
        fig_all, axes_all = plt.subplots(2, 2, figsize=(12, 10))

        saved_paths = []

        for i, (metric, color, filename) in enumerate(metrics):
            x = df[f"{metric}_gt"]
            y = df[f"{metric}_pred"]
            max_val = max(x.max(), y.max()) * 1.05

            # single plot
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(x, y, color=color, s=10, alpha=0.7)
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
            ax.set_xlabel(f"Ground Truth {metric}")
            ax.set_ylabel(f"Prediction {metric}")
            ax.set_xlim([0, max_val])
            ax.set_ylim([0, max_val])
            ax.set_title(metric)
            ax.grid(True)
            fig.tight_layout()
            single_path = os.path.join(output_dir, filename)
            fig.savefig(single_path)
            plt.close(fig)
            saved_paths.append(single_path)

            # combined plot
            ax_all = axes_all.flat[i]
            ax_all.scatter(x, y, color=color, s=10, alpha=0.7)
            ax_all.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
            ax_all.set_xlabel(f"GT {metric}")
            ax_all.set_ylabel(f"Pred {metric}")
            ax_all.set_xlim([0, max_val])
            ax_all.set_ylim([0, max_val])
            ax_all.set_title(f"({chr(97+i)}) {metric}")
            ax_all.grid(True)

        fig_all.tight_layout()
        fig_all.savefig(combined_path)
        plt.close(fig_all)
        saved_paths.append(combined_path)

        return {
            "status": "success",
            "summary": f"Figures saved to {output_dir}",
            "outputs": {
                "individual_plots": saved_paths[:-1],
                "combined_plot": combined_path
            },
            "error": None
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": "Plotting failed",
            "outputs": None,
            "error": str(e)
        }
