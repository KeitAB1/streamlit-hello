import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置收敛数据文件夹路径
convergence_data_dir = "result/ConvergenceData"

# 获取可用的优化算法收敛数据
available_algorithms = [f for f in os.listdir(convergence_data_dir) if f.endswith('.csv')]


# 从文件名中提取算法名称，确保组合算法（如 pso_sa）被正确识别
def extract_algorithm_name(file_name):
    base_name = os.path.basename(file_name)
    if "psosa" in base_name:
        return "pso_sa"
    elif "ga" in base_name:
        return "ga"
    elif "pso" in base_name:
        return "pso"
    elif "sa" in base_name:
        return "sa"
    elif "aco" in base_name:
        return "aco"
    else:
        return "unknown"


# 获取算法名称列表
algorithm_names = [extract_algorithm_name(f) for f in available_algorithms]


# 创建函数来检查文件是否包含期望的列，并处理错误
def load_convergence_data(file_path, algorithm_name):
    try:
        df = pd.read_csv(file_path)

        # 处理不同算法使用的列名
        if 'Iteration' in df.columns:
            iteration_column = 'Iteration'
        elif 'Generation' in df.columns:
            iteration_column = 'Generation'
        else:
            st.warning(
                f"The file {os.path.basename(file_path)} does not contain the required 'Iteration' or 'Generation' columns.")
            return None, None

        if 'Best Score' not in df.columns:
            st.warning(f"The file {os.path.basename(file_path)} does not contain the required 'Best Score' column.")
            return None, None

        return df, iteration_column
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None, None


# 检查每个算法的收敛数据
convergence_data = {}
iteration_columns = {}  # 记录每个算法对应的轮数列名
for file_name in available_algorithms:
    algorithm_name = extract_algorithm_name(file_name)
    file_path = os.path.join(convergence_data_dir, file_name)
    df, iteration_column = load_convergence_data(file_path, algorithm_name)
    if df is not None:
        convergence_data[algorithm_name] = df
        iteration_columns[algorithm_name] = iteration_column

# 如果没有找到有效的收敛数据，显示错误
if not convergence_data:
    st.error("No valid convergence data found. Please check your CSV files.")
else:
    # 获取不同算法的最大轮数
    max_iterations = {algo: df[iteration_columns[algo]].max() for algo, df in convergence_data.items()}

    # 确保所有算法的 max_iterations 中没有 None 或空值
    valid_max_iterations = [v for v in max_iterations.values() if v is not None and pd.notna(v)]

    if not valid_max_iterations:
        st.error("No valid iteration data found in the convergence files.")
    else:
        # 用户界面 - 左侧栏选择
        st.sidebar.title("Convergence Curve Settings")

        # 用户选择要比较的算法，默认只选中 SA
        selected_algorithms = st.sidebar.multiselect("Select Algorithms for Comparison", algorithm_names, default=["sa"])

        # 用户选择最大显示轮数，默认显示 10 轮
        if selected_algorithms:
            max_iter_display = st.sidebar.slider("Max Iterations to Display", min_value=1,
                                                 max_value=max(valid_max_iterations), value=10)

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
                iterations = convergence_data[algo][iteration_columns[algo]][:max_iter_display]
                best_scores = convergence_data[algo]['Best Score'][:max_iter_display]

                # 检查是否有足够的轮数数据
                if not best_scores.empty:
                    plt.plot(iterations, best_scores, label=f"{algo} (Max {len(iterations)} {iteration_columns[algo]})")

                    # 记录最终的 Best Score，确保数据非空
                    final_best_scores[algo] = best_scores.iloc[-1]
                else:
                    final_best_scores[algo] = "No data"

            plt.xlabel("Iterations / Generations")
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
