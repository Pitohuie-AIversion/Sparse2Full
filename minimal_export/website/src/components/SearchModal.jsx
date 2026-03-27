import React, { useEffect, useRef, useState } from 'react';
import { Search, X } from 'lucide-react';
import Fuse from 'fuse.js';

// Mock data for search
const SEARCH_DATA = [
  { title: '项目概述', id: 'overview', content: 'Sparse2Full 是一个先进的深度学习框架...' },
  { title: '核心特性', id: 'features', content: '支持实时渲染、高精度模型训练、自动调优...' },
  { title: '架构与技术栈', id: 'architecture', content: '使用 PyTorch, React, Vite, Tailwind CSS...' },
  { title: '部署指南', id: 'deployment', content: '如何配置 GitHub Pages 并通过 Actions 进行自动部署...' },
  { title: '分支管理策略', id: 'branch-strategy', content: 'main 分支作为开发分支，gh-pages 分支作为部署分支...' },
  { title: '使用示例', id: 'usage', content: '运行 train_real_data_ar.py 开始训练...' },
  { title: '常见问题解答', id: 'faq', content: '常见错误 ENOSPC 处理方法...' },
];

export default function SearchModal({ isOpen, onClose }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const inputRef = useRef(null);

  const fuse = new Fuse(SEARCH_DATA, {
    keys: ['title', 'content'],
    threshold: 0.3
  });

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
      setQuery('');
      setResults([]);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        isOpen ? onClose() : onClose(); // handled by parent, here we just close if open
      }
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const handleSearch = (e) => {
    const val = e.target.value;
    setQuery(val);
    if (val) {
      setResults(fuse.search(val).map(res => res.item));
    } else {
      setResults([]);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-16 sm:pt-24">
      <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm transition-opacity" onClick={onClose} />
      <div className="relative w-full max-w-xl transform overflow-hidden rounded-xl bg-slate-800 shadow-2xl ring-1 ring-slate-700/50 mx-4 sm:mx-0">
        <div className="flex items-center border-b border-slate-700 px-4">
          <Search className="h-5 w-5 text-slate-400" />
          <input
            ref={inputRef}
            type="text"
            className="h-14 w-full bg-transparent border-0 px-4 text-slate-200 placeholder-slate-400 focus:outline-none focus:ring-0 sm:text-sm"
            placeholder="搜索文档内容..."
            value={query}
            onChange={handleSearch}
          />
          <button onClick={onClose} className="p-2 text-slate-400 hover:text-slate-300">
            <X className="h-5 w-5" />
          </button>
        </div>
        
        {results.length > 0 && (
          <ul className="max-h-96 overflow-y-auto p-2 text-sm text-slate-300">
            {results.map((item) => (
              <li key={item.id}>
                <a
                  href={`#${item.id}`}
                  onClick={onClose}
                  className="block rounded-lg px-4 py-3 hover:bg-slate-700/50 transition-colors"
                >
                  <div className="font-medium text-slate-100">{item.title}</div>
                  <div className="text-slate-400 text-xs mt-1 truncate">{item.content}</div>
                </a>
              </li>
            ))}
          </ul>
        )}
        
        {query && results.length === 0 && (
          <div className="p-10 text-center text-sm text-slate-400">
            没有找到关于 "{query}" 的内容。
          </div>
        )}
      </div>
    </div>
  );
}