import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Search, GitBranch, MonitorPlay, Activity, Globe } from 'lucide-react';

export default function Navbar({ onSearchOpen }) {
  const { t, i18n } = useTranslation();

  const toggleLang = () => {
    i18n.changeLanguage(i18n.language === 'zh' ? 'en' : 'zh');
  };

  return (
    <nav className="sticky top-0 z-50 w-full backdrop-blur flex-none border-b border-secondary/50 bg-background/80 transition-colors duration-500" aria-label="Main Navigation">
      <div className="max-w-8xl mx-auto">
        <div className="py-4 border-b border-slate-900/10 lg:px-8 lg:border-0 dark:border-slate-300/10 mx-4 lg:mx-0">
          <div className="relative flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link to="/" className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent" aria-label="Sparse2Full Home">
                Sparse2Full
              </Link>
              <span className="hidden sm:flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" aria-label="Deployment Status">
                <Activity className="w-3 h-3 mr-1" aria-hidden="true" />
                {t('navbar.deployed')}
              </span>
            </div>

            <div className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-300">
              <Link to="/" className="hover:text-primary transition-colors">{t('navbar.home')}</Link>
              <Link to="/docs" className="hover:text-primary transition-colors">{t('navbar.docs')}</Link>
              <Link to="/about" className="hover:text-primary transition-colors">{t('navbar.about')}</Link>
              <Link to="/contact" className="hover:text-primary transition-colors">{t('navbar.contact')}</Link>
            </div>

            <div className="relative flex items-center gap-4">
              <button 
                onClick={onSearchOpen}
                className="flex items-center text-sm leading-6 text-slate-400 rounded-md ring-1 ring-slate-900/10 shadow-sm py-1.5 pl-2 pr-3 hover:ring-slate-300 dark:bg-slate-800 dark:highlight-white/5 dark:hover:bg-slate-700"
                aria-label="Open Search"
              >
                <Search className="w-4 h-4 mr-2" aria-hidden="true" />
                <span className="hidden sm:inline">{t('navbar.search')}</span>
                <span className="sm:hidden">{t('navbar.search_mobile')}</span>
                <span className="ml-auto pl-3 flex-none text-xs font-semibold">⌘K</span>
              </button>

              <div className="flex items-center gap-4 border-l border-slate-200 ml-4 pl-4 dark:border-slate-800">
                <button onClick={toggleLang} className="text-slate-400 hover:text-slate-50 transition-colors" aria-label="Toggle Language">
                  <Globe className="w-5 h-5" />
                </button>
                <Link to="/login" className="text-sm font-medium text-slate-300 hover:text-primary transition-colors">
                  {t('navbar.login')}
                </Link>
                <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-slate-50 transition-colors" aria-label="GitHub Repository">
                  <GitBranch className="w-5 h-5" aria-hidden="true" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}