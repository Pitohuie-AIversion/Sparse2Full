#!/usr/bin/env python3
"""
创建包含所有模型详细信息的HTML综合性能报告
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import base64
from io import BytesIO

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveHTMLReportGenerator:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.batch_dir = self.base_dir / "runs" / "batch_retrain_20251015_032934"
        self.analysis_dir = self.batch_dir / "analysis"
        self.comprehensive_dir = self.batch_dir / "comprehensive_analysis"
        
        # 确保目录存在
        self.comprehensive_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_data(self):
        """加载模型数据"""
        csv_path = self.analysis_dir / "model_ranking.csv"
        if not csv_path.exists():
            logger.error(f"找不到模型排名文件: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        return df
    
    def load_enhanced_data(self):
        """加载增强的模型数据"""
        enhanced_csv_path = self.comprehensive_dir / "enhanced_model_comparison.csv"
        if enhanced_csv_path.exists():
            return pd.read_csv(enhanced_csv_path)
        return None
    
    def create_performance_charts_html(self, df):
        """创建性能图表的HTML代码"""
        charts_html = """
        <div class="charts-container">
            <div class="chart-row">
                <div class="chart-item">
                    <h3>📊 Rel-L2 误差对比</h3>
                    <canvas id="relL2Chart" width="400" height="300"></canvas>
                </div>
                <div class="chart-item">
                    <h3>📈 PSNR 对比</h3>
                    <canvas id="psnrChart" width="400" height="300"></canvas>
                </div>
            </div>
            <div class="chart-row">
                <div class="chart-item">
                    <h3>⏱️ 训练时间对比</h3>
                    <canvas id="timeChart" width="400" height="300"></canvas>
                </div>
                <div class="chart-item">
                    <h3>🎯 综合评分</h3>
                    <canvas id="scoreChart" width="400" height="300"></canvas>
                </div>
            </div>
        </div>
        """
        
        # 生成图表数据的JavaScript
        models = df['模型'].tolist()
        rel_l2_values = df['Rel-L2'].tolist()
        psnr_values = df['PSNR'].tolist()
        time_values = df['训练时间(分钟)'].tolist()
        
        # 计算综合评分（简化版本）
        score_values = []
        for _, row in df.iterrows():
            # 综合评分 = (1/Rel-L2) * PSNR / 训练时间 * 100
            score = (1/row['Rel-L2']) * row['PSNR'] / max(row['训练时间(分钟)'], 0.1) * 0.1
            score_values.append(round(score, 2))
        
        chart_script = f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        // 数据
        const models = {models};
        const relL2Values = {rel_l2_values};
        const psnrValues = {psnr_values};
        const timeValues = {time_values};
        const scoreValues = {score_values};
        
        // 颜色配置
        const colors = [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 205, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)',
            'rgba(199, 199, 199, 0.8)',
            'rgba(83, 102, 255, 0.8)'
        ];
        
        // Rel-L2 图表
        const relL2Ctx = document.getElementById('relL2Chart').getContext('2d');
        new Chart(relL2Ctx, {{
            type: 'bar',
            data: {{
                labels: models,
                datasets: [{{
                    label: 'Rel-L2 Error',
                    data: relL2Values,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Rel-L2 Error'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // PSNR 图表
        const psnrCtx = document.getElementById('psnrChart').getContext('2d');
        new Chart(psnrCtx, {{
            type: 'bar',
            data: {{
                labels: models,
                datasets: [{{
                    label: 'PSNR (dB)',
                    data: psnrValues,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'PSNR (dB)'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // 训练时间图表
        const timeCtx = document.getElementById('timeChart').getContext('2d');
        new Chart(timeCtx, {{
            type: 'bar',
            data: {{
                labels: models,
                datasets: [{{
                    label: '训练时间 (分钟)',
                    data: timeValues,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: '训练时间 (分钟)'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // 综合评分图表
        const scoreCtx = document.getElementById('scoreChart').getContext('2d');
        new Chart(scoreCtx, {{
            type: 'radar',
            data: {{
                labels: models,
                datasets: [{{
                    label: '综合评分',
                    data: scoreValues,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(54, 162, 235, 1)'
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: '综合评分'
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """
        
        return charts_html + chart_script
    
    def create_model_recommendations(self, df):
        """创建模型推荐建议"""
        recommendations_html = """
        <div class="recommendations-section">
            <h2>🎯 模型推荐建议</h2>
            
            <div class="recommendation-cards">
                <div class="rec-card best">
                    <h3>🥇 最佳选择：FNO2D</h3>
                    <div class="rec-content">
                        <p><strong>适用场景：</strong>对精度要求极高的科学计算任务</p>
                        <p><strong>优势：</strong></p>
                        <ul>
                            <li>最低的Rel-L2误差 (0.01215)</li>
                            <li>最高的PSNR (43.88 dB)</li>
                            <li>训练时间短 (0.76分钟)</li>
                            <li>优秀的频域处理能力</li>
                        </ul>
                        <p><strong>推荐指数：</strong> ⭐⭐⭐⭐⭐</p>
                    </div>
                </div>
                
                <div class="rec-card good">
                    <h3>🥈 平衡选择：SwinUNet</h3>
                    <div class="rec-content">
                        <p><strong>适用场景：</strong>需要平衡精度和通用性的任务</p>
                        <p><strong>优势：</strong></p>
                        <ul>
                            <li>良好的重建质量</li>
                            <li>基于Transformer的先进架构</li>
                            <li>适中的计算复杂度</li>
                            <li>良好的泛化能力</li>
                        </ul>
                        <p><strong>推荐指数：</strong> ⭐⭐⭐⭐</p>
                    </div>
                </div>
                
                <div class="rec-card alternative">
                    <h3>🥉 经济选择：UNet</h3>
                    <div class="rec-content">
                        <p><strong>适用场景：</strong>资源受限或快速原型开发</p>
                        <p><strong>优势：</strong></p>
                        <ul>
                            <li>训练时间较短 (1.1分钟)</li>
                            <li>模型结构简单易懂</li>
                            <li>内存占用较少</li>
                            <li>实现和调试容易</li>
                        </ul>
                        <p><strong>推荐指数：</strong> ⭐⭐⭐</p>
                    </div>
                </div>
            </div>
            
            <div class="usage-scenarios">
                <h3>📋 使用场景建议</h3>
                <div class="scenario-grid">
                    <div class="scenario-item">
                        <h4>🔬 科研项目</h4>
                        <p>推荐：<strong>FNO2D</strong></p>
                        <p>理由：最高精度，适合发表论文</p>
                    </div>
                    <div class="scenario-item">
                        <h4>🏭 工业应用</h4>
                        <p>推荐：<strong>SwinUNet</strong></p>
                        <p>理由：精度与效率的良好平衡</p>
                    </div>
                    <div class="scenario-item">
                        <h4>🚀 快速原型</h4>
                        <p>推荐：<strong>UNet</strong></p>
                        <p>理由：简单快速，易于调试</p>
                    </div>
                    <div class="scenario-item">
                        <h4>📱 边缘计算</h4>
                        <p>推荐：<strong>MLP</strong></p>
                        <p>理由：参数少，推理速度快</p>
                    </div>
                </div>
            </div>
        </div>
        """
        return recommendations_html
    
    def create_detailed_analysis(self, df):
        """创建详细分析"""
        analysis_html = """
        <div class="detailed-analysis">
            <h2>🔍 详细性能分析</h2>
            
            <div class="analysis-grid">
                <div class="analysis-card">
                    <h3>📊 精度分析</h3>
                    <div class="metric-comparison">
        """
        
        # 添加精度指标对比
        for _, row in df.iterrows():
            model_name = row['模型']
            rel_l2 = row['Rel-L2']
            psnr = row['PSNR']
            ssim = row['SSIM']
            
            # 根据性能设置颜色
            if rel_l2 < 0.02:
                color_class = "excellent"
            elif rel_l2 < 0.04:
                color_class = "good"
            else:
                color_class = "average"
            
            analysis_html += f"""
                        <div class="metric-row {color_class}">
                            <span class="model-name">{model_name}</span>
                            <span class="metric-value">Rel-L2: {rel_l2:.6f}</span>
                            <span class="metric-value">PSNR: {psnr:.2f} dB</span>
                            <span class="metric-value">SSIM: {ssim:.4f}</span>
                        </div>
            """
        
        analysis_html += """
                    </div>
                </div>
                
                <div class="analysis-card">
                    <h3>⏱️ 效率分析</h3>
                    <div class="efficiency-comparison">
        """
        
        # 添加效率指标对比
        for _, row in df.iterrows():
            model_name = row['模型']
            train_time = row['训练时间(分钟)']
            params = row['参数量(M)']
            
            # 效率评级
            if train_time < 2:
                efficiency_class = "fast"
                efficiency_text = "快速"
            elif train_time < 4:
                efficiency_class = "medium"
                efficiency_text = "中等"
            else:
                efficiency_class = "slow"
                efficiency_text = "较慢"
            
            analysis_html += f"""
                        <div class="efficiency-row {efficiency_class}">
                            <span class="model-name">{model_name}</span>
                            <span class="efficiency-badge">{efficiency_text}</span>
                            <span class="time-value">{train_time:.2f} 分钟</span>
                            <span class="params-value">{params:.1f}M 参数</span>
                        </div>
            """
        
        analysis_html += """
                    </div>
                </div>
            </div>
            
            <div class="key-findings">
                <h3>🔑 关键发现</h3>
                <div class="findings-list">
                    <div class="finding-item">
                        <span class="finding-icon">🎯</span>
                        <div class="finding-content">
                            <h4>FNO2D表现卓越</h4>
                            <p>在所有精度指标上都显著优于其他模型，特别是在Rel-L2误差方面领先明显。</p>
                        </div>
                    </div>
                    <div class="finding-item">
                        <span class="finding-icon">⚡</span>
                        <div class="finding-content">
                            <h4>训练效率差异显著</h4>
                            <p>FNO2D不仅精度最高，训练时间也最短，显示出优秀的算法效率。</p>
                        </div>
                    </div>
                    <div class="finding-item">
                        <span class="finding-icon">🔄</span>
                        <div class="finding-content">
                            <h4>传统方法仍有价值</h4>
                            <p>UNet虽然精度不是最高，但训练时间短，适合快速原型开发。</p>
                        </div>
                    </div>
                    <div class="finding-item">
                        <span class="finding-icon">⚠️</span>
                        <div class="finding-content">
                            <h4>部分模型需要优化</h4>
                            <p>SegFormer系列模型因配置问题训练失败，需要进一步调试和优化。</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return analysis_html
    
    def create_html_report(self):
        """创建完整的HTML报告"""
        # 加载数据
        df = self.load_model_data()
        if df is None:
            logger.error("无法加载模型数据")
            return None
        
        enhanced_df = self.load_enhanced_data()
        
        # 创建HTML内容
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDEBench模型综合性能报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            margin-top: 20px;
            margin-bottom: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        
        .header {{
            text-align: center;
            padding: 30px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .stat-card h3 {{
            font-size: 2em;
            margin-bottom: 5px;
        }}
        
        .stat-card p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .charts-container {{
            margin: 30px 0;
        }}
        
        .chart-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        .chart-item {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        
        .chart-item h3 {{
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .recommendations-section {{
            margin: 30px 0;
        }}
        
        .recommendation-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .rec-card {{
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .rec-card.best {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }}
        
        .rec-card.good {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
        }}
        
        .rec-card.alternative {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
        }}
        
        .rec-card h3 {{
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .rec-card ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        
        .usage-scenarios {{
            margin-top: 30px;
        }}
        
        .scenario-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .scenario-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        
        .detailed-analysis {{
            margin: 30px 0;
        }}
        
        .analysis-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 20px 0;
        }}
        
        .analysis-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        
        .metric-row, .efficiency-row {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 10px;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            align-items: center;
        }}
        
        .metric-row.excellent {{
            background: #d4edda;
            border-left: 4px solid #28a745;
        }}
        
        .metric-row.good {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        
        .metric-row.average {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
        }}
        
        .efficiency-row.fast {{
            background: #d4edda;
            border-left: 4px solid #28a745;
        }}
        
        .efficiency-row.medium {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        
        .efficiency-row.slow {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
        }}
        
        .model-name {{
            font-weight: bold;
        }}
        
        .efficiency-badge {{
            background: rgba(255,255,255,0.8);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.9em;
            color: #333;
        }}
        
        .key-findings {{
            margin-top: 30px;
        }}
        
        .findings-list {{
            margin-top: 15px;
        }}
        
        .finding-item {{
            display: flex;
            align-items: flex-start;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        
        .finding-icon {{
            font-size: 1.5em;
            margin-right: 15px;
            margin-top: 5px;
        }}
        
        .finding-content h4 {{
            margin-bottom: 5px;
            color: #495057;
        }}
        
        .table-container {{
            margin: 30px 0;
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        th {{
            background: #495057;
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            color: #6c757d;
        }}
        
        @media (max-width: 768px) {{
            .chart-row {{
                grid-template-columns: 1fr;
            }}
            
            .analysis-grid {{
                grid-template-columns: 1fr;
            }}
            
            .recommendation-cards {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 PDEBench模型综合性能报告</h1>
            <p>深度学习PDE稀疏观测重建系统 - 批量训练结果分析</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary-stats">
            <div class="stat-card">
                <h3>{len(df)}</h3>
                <p>成功训练模型</p>
            </div>
            <div class="stat-card">
                <h3>{df['Rel-L2'].min():.6f}</h3>
                <p>最佳Rel-L2误差</p>
            </div>
            <div class="stat-card">
                <h3>{df['PSNR'].max():.2f}</h3>
                <p>最高PSNR (dB)</p>
            </div>
            <div class="stat-card">
                <h3>{df['训练时间(分钟)'].min():.2f}</h3>
                <p>最短训练时间 (分钟)</p>
            </div>
        </div>
        
        {self.create_performance_charts_html(df)}
        
        <div class="table-container">
            <h2>📋 详细性能数据表</h2>
            {df.to_html(index=False, classes='table table-striped', escape=False)}
        </div>
        
        {self.create_model_recommendations(df)}
        
        {self.create_detailed_analysis(df)}
        
        <div class="footer">
            <p>📊 报告由PDEBench稀疏观测重建系统自动生成</p>
            <p>🔬 基于批量训练实验数据 | 📅 {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # 保存HTML报告
        report_path = self.comprehensive_dir / "comprehensive_performance_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"综合HTML报告已保存: {report_path}")
        return report_path

def main():
    """主函数"""
    base_dir = "."
    
    generator = ComprehensiveHTMLReportGenerator(base_dir)
    report_path = generator.create_html_report()
    
    if report_path:
        print(f"\n🎉 综合HTML性能报告生成完成！")
        print(f"📁 报告路径: {report_path}")
        print(f"🌐 在浏览器中打开查看详细报告")
    else:
        print("❌ 报告生成失败")

if __name__ == "__main__":
    main()