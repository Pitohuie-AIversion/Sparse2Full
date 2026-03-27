import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function Docs() {
  const yamlCode = `name: Deploy to GitHub Pages

on:
  push:
    branches: ["main"]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: 18

      - name: Install and Build
        run: |
          cd website
          npm ci
          npm run build

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: \${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./website/dist
          publish_branch: gh-pages`;
  return (
    <div className="prose prose-invert prose-slate max-w-none">
      <section id="overview" className="mb-24 scroll-mt-24">
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-200 sm:text-5xl mb-6">
          Sparse2Full Docs
        </h1>
        <p className="text-xl text-slate-400 mb-8">
          先进的深度学习框架与工具链，致力于提供从稀疏数据到完整重建的高效解决方案。本文档展示了我们如何通过 GitHub Pages 和 deploy from branch 策略实现持续部署。
        </p>
      </section>

      <section id="features" className="mb-24 scroll-mt-24 border-t border-slate-800/60 pt-10">
        <h2 className="text-3xl font-bold mb-8">核心特性</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            { title: '自动化部署', desc: '基于 GitHub Actions 结合 deploy from branch 策略，实现文档代码一键发布。' },
            { title: '高性能训练', desc: '优化底层算子，支持多节点分布式训练，提高数据吞吐量。' },
            { title: '模块化设计', desc: '核心功能拆分为独立模块，便于二次开发与集成扩展。' },
            { title: '现代化文档', desc: '基于 React + Tailwind CSS 打造的沉浸式极客风文档站点，支持全局检索。' }
          ].map((f, i) => (
            <div key={i} className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50 hover:border-primary/50 transition-colors">
              <h3 className="text-xl font-semibold text-slate-200 mt-0">{f.title}</h3>
              <p className="text-slate-400 mb-0">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="architecture" className="mb-24 scroll-mt-24 border-t border-slate-800/60 pt-10">
        <h2 className="text-3xl font-bold mb-8">架构与技术栈</h2>
        <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50 mb-8">
          <h3 className="text-xl font-semibold mb-4 flex items-center"><span className="w-2 h-2 rounded-full bg-blue-400 mr-3"></span>前端展示</h3>
          <p className="text-slate-400">基于 <strong className="text-slate-200">React 19</strong> 与 <strong className="text-slate-200">Vite</strong> 构建，使用 <strong className="text-slate-200">Tailwind CSS</strong> 进行响应式原子化样式开发。集成了 <strong className="text-slate-200">lucide-react</strong> 图标库与 <strong className="text-slate-200">fuse.js</strong> 模糊搜索，提供流畅的极客风阅读体验。</p>
        </div>
        <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50 mb-8">
          <h3 className="text-xl font-semibold mb-4 flex items-center"><span className="w-2 h-2 rounded-full bg-green-400 mr-3"></span>部署流水线</h3>
          <p className="text-slate-400">使用 <strong className="text-slate-200">GitHub Actions</strong> 自动化工作流。每当主分支（<code className="bg-slate-800 px-1 py-0.5 rounded text-sm text-pink-400">main</code>）有代码推送时，自动触发构建并将产物推送到 <code className="bg-slate-800 px-1 py-0.5 rounded text-sm text-pink-400">gh-pages</code> 分支，由 GitHub Pages 接管并对外发布静态站点。</p>
        </div>
        
        <h3 className="text-2xl font-bold mt-12 mb-6">项目结构</h3>
        <div className="bg-[#0d1117] rounded-xl p-6 overflow-x-auto font-mono text-sm border border-slate-700/50 text-slate-300">
<pre><code>{`Sparse2Full/
├── .github/
│   └── workflows/
│       └── deploy.yml      # GitHub Actions 部署配置文件
├── tools/
│   └── training/
│       └── train_real_data_ar.py # 核心训练脚本
├── website/                # 静态文档站点前端源码
│   ├── src/
│   │   ├── components/     # React 组件
│   │   ├── App.jsx         # 页面主体
│   │   └── main.jsx        # 前端入口
│   ├── index.html          # HTML 模板
│   ├── package.json        # 前端依赖配置
│   ├── tailwind.config.js  # 样式主题配置
│   └── vite.config.js      # 构建配置
└── README.md               # 项目自述文件`}</code></pre>
        </div>
      </section>

      <section id="deployment" className="mb-24 scroll-mt-24 border-t border-slate-800/60 pt-10">
        <h2 className="text-3xl font-bold mb-8">部署指南</h2>
        <div className="space-y-6">
          <h3 className="text-xl font-semibold text-slate-200">1. GitHub Pages 配置</h3>
          <p className="text-slate-400">
            在 GitHub 仓库中，进入 <strong>Settings &gt; Pages</strong>。在 "Build and deployment" 下，将 Source 设置为 <strong>Deploy from a branch</strong>，并选择 <code>gh-pages</code> 分支的 <code>/(root)</code> 目录。
          </p>

          <h3 className="text-xl font-semibold text-slate-200 mt-8">2. GitHub Actions 工作流</h3>
          <p className="text-slate-400">
            项目使用 GitHub Actions 自动构建 React 静态站点，并将编译后的产物推送到 <code>gh-pages</code> 分支。以下是 <code>.github/workflows/deploy.yml</code> 的配置内容：
          </p>
          <div className="rounded-xl overflow-hidden border border-slate-700/50 my-4 shadow-xl">
            <SyntaxHighlighter language="yaml" style={vscDarkPlus} customStyle={{ margin: 0, padding: '1.5rem', background: '#0d1117' }}>
              {yamlCode}
            </SyntaxHighlighter>
          </div>
        </div>
      </section>

      <section id="branch-strategy" className="mb-24 scroll-mt-24 border-t border-slate-800/60 pt-10">
        <h2 className="text-3xl font-bold mb-8">分支管理策略</h2>
        <div className="flex flex-col md:flex-row gap-6">
          <div className="flex-1 bg-slate-900/50 p-6 rounded-2xl border border-slate-700/50">
            <h3 className="text-xl font-semibold text-blue-400 mb-3">main (开发分支)</h3>
            <p className="text-slate-400">
              存储所有的源代码，包括 Python 核心框架代码和 <code>website/</code> 目录下的 React 前端源码。所有的 PR 和功能开发都在此分支进行。
            </p>
          </div>
          <div className="hidden md:flex items-center justify-center text-slate-600">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m9 18 6-6-6-6"/></svg>
          </div>
          <div className="flex-1 bg-slate-900/50 p-6 rounded-2xl border border-slate-700/50">
            <h3 className="text-xl font-semibold text-emerald-400 mb-3">gh-pages (部署分支)</h3>
            <p className="text-slate-400">
              孤儿分支（Orphan branch），仅存储由 <code>main</code> 分支构建生成的静态前端产物（<code>dist/</code> 目录内容）。不包含任何源代码，专门用于 GitHub Pages 托管。
            </p>
          </div>
        </div>
      </section>

      <section id="usage" className="mb-24 scroll-mt-24 border-t border-slate-800/60 pt-10">
        <h2 className="text-3xl font-bold mb-8">使用示例</h2>
        <p className="text-slate-400 mb-4">通过以下命令即可启动数据训练任务：</p>
        <div className="bg-[#0d1117] rounded-xl p-4 overflow-x-auto font-mono text-sm border border-slate-700/50 text-blue-400 mb-8">
          <code>python tools/training/train_real_data_ar.py --config configs/default.yaml</code>
        </div>
      </section>

      <section id="faq" className="mb-24 scroll-mt-24 border-t border-slate-800/60 pt-10">
        <h2 className="text-3xl font-bold mb-8">常见问题解答</h2>
        <div className="space-y-4">
          <details className="group bg-slate-900/50 rounded-xl border border-slate-700/50 [&_summary::-webkit-details-marker]:hidden">
            <summary className="flex cursor-pointer items-center justify-between p-6 text-slate-200 font-medium">
              部署时提示 Permission Denied 怎么办？
              <span className="relative ml-1.5 h-5 w-5 flex-shrink-0">
                <svg className="absolute inset-0 h-5 w-5 opacity-100 group-open:opacity-0 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" /></svg>
                <svg className="absolute inset-0 h-5 w-5 opacity-0 group-open:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M20 12H4" /></svg>
              </span>
            </summary>
            <div className="px-6 pb-6 text-slate-400">
              请检查 GitHub Actions 的 Workflow 权限设置。进入 Settings &gt; Actions &gt; General，将 Workflow permissions 设置为 "Read and write permissions"。
            </div>
          </details>
          <details className="group bg-slate-900/50 rounded-xl border border-slate-700/50 [&_summary::-webkit-details-marker]:hidden">
            <summary className="flex cursor-pointer items-center justify-between p-6 text-slate-200 font-medium">
              Vite 启动报错 ENOSPC: System limit for number of file watchers reached
              <span className="relative ml-1.5 h-5 w-5 flex-shrink-0">
                <svg className="absolute inset-0 h-5 w-5 opacity-100 group-open:opacity-0 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" /></svg>
                <svg className="absolute inset-0 h-5 w-5 opacity-0 group-open:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M20 12H4" /></svg>
              </span>
            </summary>
            <div className="px-6 pb-6 text-slate-400">
              这是 Linux 系统的文件监听限制导致的。可通过执行 <code>echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p</code> 解决。
            </div>
          </details>
        </div>
      </section>

      <section id="community" className="mb-24 scroll-mt-24 border-t border-slate-800/60 pt-10">
        <h2 className="text-3xl font-bold mb-8">社区与支持</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
          <div>
            <h3 className="text-xl font-semibold mb-4 text-slate-200">版本更新日志 (Changelog)</h3>
            <ul className="list-disc pl-5 text-slate-400 space-y-2">
              <li><strong className="text-slate-300">v1.1.0</strong>: 引入基于 React 的自动化部署文档</li>
              <li><strong className="text-slate-300">v1.0.0</strong>: 初始版本，完成核心算子开发</li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4 text-slate-200">贡献指南</h3>
            <p className="text-slate-400 mb-4">欢迎提交 Pull Request！请确保在提交前运行了所有的单元测试，并遵循项目的代码规范。</p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4 text-slate-200">许可证</h3>
            <p className="text-slate-400">本项目采用 <a href="#" className="text-primary hover:underline">MIT License</a> 进行开源。</p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4 text-slate-200">联系方式</h3>
            <p className="text-slate-400">
              如果在运行或部署中遇到问题，请通过 <a href="https://github.com" className="text-primary hover:underline">GitHub Issues</a> 与我们取得联系。
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}