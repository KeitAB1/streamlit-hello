import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置收敛数据文件夹路径
convergence_data_dir = "result/ConvergenceData"

# 获取可用的优化算法收敛数据
available_algorithms = [f for f in os.listdir(convergence_data_dir) if f.endswith('.csv')]

# 从文件名中提取算法名称
algorithm_names = [f.split('_')[2].split('.')[0] for f in available_algorithms]


# 创建函数来检查文件是否包含期望的列，并处理错误
def load_convergence_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Iteration' not in df.columns or 'Best Score' not in df.columns:
            st.warning(
                f"The file {os.path.basename(file_path)} does not contain the required 'Iteration' or 'Best Score' columns.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None


# 检查每个算法的收敛数据
convergence_data = {}
for file_name in available_algorithms:
    algorithm_name = file_name.split('_')[2].split('.')[0]
    file_path = os.path.join(convergence_data_dir, file_name)
    df = load_convergence_data(file_path)
    if df is not None:
        convergence_data[algorithm_name] = df

# 如果没有找到有效的收敛数据，显示错误
if not convergence_data:
    st.error("No valid convergence data found. Please check your CSV files.")
else:
    # 获取不同算法的最大轮数
    max_iterations = {algo: len(df) for algo, df in convergence_data.items()}

    # 用户界面 - 左侧栏选择
    st.sidebar.title("Convergence Curve Settings")

    # 用户选择要比较的算法
    selected_algorithms = st.sidebar.multiselect("Select Algorithms for Comparison", algorithm_names,
                                                 default=algorithm_names)

    # 用户选择最大显示轮数
    if selected_algorithms:
        max_iter_display = st.sidebar.slider("Max Iterations to Display", min_value=1,
                                             max_value=max(max_iterations.values()), value=min(max_iterations.values()))

        # 检查是否有算法的轮数不同并提示用户
        min_iter = min([max_iterations[algo] for algo in selected_algorithms])
        max_iter = max([max_iterations[algo] for algo in selected_algorithms])
        if min_iter != max_iter:
            st.sidebar.warning(
                f"Some algorithms have fewer iterations. Please run the following algorithms for {max_iter - min_iter} more iterations to match the highest iterations: "
                f"{[algo for algo in selected_algorithms if max_iterations[algo] < max_iter]}")

        # 绘制收敛曲线
        st.title("Convergence Curves Comparison")

        plt.figure(figsize=(10, 6))

        final_best_scores = {}

        for algo in selected_algorithms:
            iterations = convergence_data[algo]['Iteration'][:max_iter_display]
            best_scores = convergence_data[algo]['Best Score'][:max_iter_display]
            plt.plot(iterations, best_scores, label=f"{algo} (Max {len(iterations)} iterations)")

            # 记录最终的 Best Score
            final_best_scores[algo] = best_scores.iloc[-1]  # 获取最后一轮的 Best Score

        plt.xlabel("Iterations")
        plt.ylabel("Best Score")
        plt.title("Convergence Curves")
        plt.legend()
        plt.grid(True)

        # 显示图表
        st.pyplot(plt)

        # 显示最终对比的 Best Score
        st.subheader("Final Best Scores Comparison")

        # 创建一个 DataFrame 显示最终 Best Score
        final_scores_df = pd.DataFrame({
            'Algorithm': list(final_best_scores.keys()),
            'Final Best Score': list(final_best_scores.values())
        })

        # 显示 DataFrame
        st.table(final_scores_df)
    else:
        st.warning("Please select at least one algorithm to compare.")
