import React from 'react';
import { useTranslation } from 'react-i18next';

export default function About() {
  const { t } = useTranslation();

  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6">
      <h1 className="text-4xl font-bold text-slate-200 mb-8">{t('navbar.about')}</h1>
      <div className="prose prose-invert prose-slate max-w-none">
        <p className="text-lg text-slate-400">
          我们是一个致力于前沿人工智能和深度学习架构研发的团队。我们的使命是提供最强大、最易用、最开放的计算工具链，让复杂的模型训练和数据处理变得简单。
        </p>
        <p className="text-lg text-slate-400">
          Sparse2Full 是我们在稀疏数据重建领域的重要成果。通过本项目，我们希望为开源社区贡献一份力量，推动该领域的发展。
        </p>
      </div>
    </div>
  );
}