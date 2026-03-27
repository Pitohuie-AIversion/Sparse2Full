import React from 'react';
import { useTranslation } from 'react-i18next';

export default function Contact() {
  const { t } = useTranslation();

  return (
    <div className="max-w-2xl mx-auto py-12 px-4 sm:px-6">
      <h1 className="text-4xl font-bold text-slate-200 mb-8">{t('navbar.contact')}</h1>
      <form className="space-y-6 bg-slate-900/50 p-8 rounded-2xl border border-slate-800">
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-slate-300">姓名</label>
          <input type="text" id="name" aria-label="姓名" className="mt-1 block w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-200 placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary" placeholder="您的名字" />
        </div>
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-slate-300">邮箱</label>
          <input type="email" id="email" aria-label="邮箱" className="mt-1 block w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-200 placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary" placeholder="you@example.com" />
        </div>
        <div>
          <label htmlFor="message" className="block text-sm font-medium text-slate-300">留言</label>
          <textarea id="message" rows={4} aria-label="留言" className="mt-1 block w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-200 placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary" placeholder="你想对我们说的话..."></textarea>
        </div>
        <button type="button" className="w-full bg-primary hover:bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg transition-colors">
          发送消息
        </button>
      </form>
    </div>
  );
}