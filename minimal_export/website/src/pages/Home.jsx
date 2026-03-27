import React from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';

export default function Home() {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] text-center px-4">
      <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8 bg-clip-text text-transparent bg-gradient-to-r from-primary via-accent to-blue-400">
        {t('home.hero_title')}
      </h1>
      <p className="text-xl md:text-2xl text-slate-400 max-w-3xl mb-12">
        {t('home.hero_subtitle')}
      </p>
      
      <div className="flex flex-wrap gap-6 justify-center">
        <Link to="/docs" className="bg-primary hover:bg-blue-600 text-white px-8 py-4 rounded-xl font-bold transition-all transform hover:scale-105 shadow-[0_0_20px_rgba(59,130,246,0.5)]">
          {t('home.get_started')}
        </Link>
        <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-8 py-4 rounded-xl font-bold transition-all border border-slate-700">
          GitHub Repository
        </a>
      </div>

      <div className="mt-24 grid grid-cols-1 md:grid-cols-2 gap-8 text-left w-full max-w-4xl">
        <div className="p-8 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm">
          <h3 className="text-2xl font-bold text-slate-200 mb-4">{t('home.f1_title')}</h3>
          <p className="text-slate-400">{t('home.f1_desc')}</p>
        </div>
        <div className="p-8 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm">
          <h3 className="text-2xl font-bold text-slate-200 mb-4">{t('home.f2_title')}</h3>
          <p className="text-slate-400">{t('home.f2_desc')}</p>
        </div>
      </div>
    </div>
  );
}