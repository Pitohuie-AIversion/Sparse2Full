import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import SearchModal from './SearchModal';

export default function Layout({ children }) {
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('overview');
  const location = useLocation();

  const isDocs = location.pathname === '/docs';

  useEffect(() => {
    if (!isDocs) return;
    const handleScroll = () => {
      const sections = document.querySelectorAll('section[id]');
      let current = 'overview';
      sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (window.scrollY >= sectionTop - 100) {
          current = section.getAttribute('id');
        }
      });
      setActiveSection(current);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [isDocs]);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar onSearchOpen={() => setIsSearchOpen(true)} />
      <div className="flex-1 max-w-8xl mx-auto px-4 sm:px-6 md:px-8 w-full">
        {isDocs && <Sidebar activeSection={activeSection} />}
        <main className={isDocs ? "lg:pl-[19.5rem]" : ""}>
          <div className={`mx-auto pt-10 ${isDocs ? "max-w-4xl xl:max-w-none xl:ml-0 xl:mr-[15.5rem] xl:pr-16" : "max-w-6xl"}`}>
            {children}
          </div>
        </main>
      </div>
      <SearchModal isOpen={isSearchOpen} onClose={() => setIsSearchOpen(false)} />
    </div>
  );
}