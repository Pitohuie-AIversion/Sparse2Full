import React from 'react';

const SECTIONS = [
  { id: 'overview', title: '项目概述' },
  { id: 'features', title: '核心特性' },
  { id: 'architecture', title: '架构与技术栈' },
  { id: 'deployment', title: '部署指南' },
  { id: 'branch-strategy', title: '分支管理策略' },
  { id: 'usage', title: '使用示例' },
  { id: 'faq', title: '常见问题解答' },
  { id: 'community', title: '社区与支持' }
];

export default function Sidebar({ activeSection }) {
  return (
    <div className="hidden lg:block fixed z-20 inset-0 top-[3.8125rem] left-[max(0px,calc(50%-45rem))] right-auto w-[19rem] pb-10 px-8 overflow-y-auto border-r border-secondary/50 bg-background/50 backdrop-blur-sm">
      <nav className="relative lg:text-sm lg:leading-6 mt-8">
        <ul className="space-y-4">
          <li>
            <h5 className="mb-4 font-semibold text-slate-100">目录</h5>
            <ul className="space-y-3 border-l border-slate-800">
              {SECTIONS.map((section) => (
                <li key={section.id}>
                  <a 
                    href={`#${section.id}`} 
                    className={`block border-l -ml-px pl-4 ${
                      activeSection === section.id 
                        ? 'text-primary border-primary font-medium' 
                        : 'text-slate-400 border-transparent hover:border-slate-500 hover:text-slate-300'
                    }`}
                  >
                    {section.title}
                  </a>
                </li>
              ))}
            </ul>
          </li>
        </ul>
      </nav>
    </div>
  );
}