import React from 'react';
import { useTranslation } from 'react-i18next';
import { Users, Settings, Activity, Database } from 'lucide-react';

export default function Admin() {
  const { t } = useTranslation();

  const stats = [
    { name: '总访问量 (Mock)', value: '124,592', icon: Activity },
    { name: '注册用户 (Mock)', value: '8,234', icon: Users },
    { name: '数据库状态 (Mock)', value: '健康', icon: Database },
    { name: '系统负载 (Mock)', value: '23%', icon: Settings },
  ];

  return (
    <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
      <div className="md:flex md:items-center md:justify-between mb-8">
        <div className="min-w-0 flex-1">
          <h2 className="text-2xl font-bold leading-7 text-slate-200 sm:truncate sm:text-3xl sm:tracking-tight">
            后台管理看板
          </h2>
        </div>
      </div>

      <div className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((item) => (
          <div key={item.name} className="relative overflow-hidden rounded-2xl bg-slate-900/50 px-4 pt-5 pb-12 shadow sm:px-6 sm:pt-6 border border-slate-800">
            <dt>
              <div className="absolute rounded-md bg-primary/20 p-3">
                <item.icon className="h-6 w-6 text-primary" aria-hidden="true" />
              </div>
              <p className="ml-16 truncate text-sm font-medium text-slate-400">{item.name}</p>
            </dt>
            <dd className="ml-16 flex items-baseline pb-6 sm:pb-7">
              <p className="text-2xl font-semibold text-slate-200">{item.value}</p>
            </dd>
          </div>
        ))}
      </div>

      <div className="mt-8">
        <div className="rounded-2xl bg-slate-900/50 p-6 border border-slate-800">
          <h3 className="text-lg font-medium leading-6 text-slate-200 mb-4">最新系统日志</h3>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center space-x-4 text-sm">
                <span className="text-emerald-400 font-mono">[INFO]</span>
                <span className="text-slate-400 font-mono">2023-10-{10+i} 14:32:01</span>
                <span className="text-slate-300">系统服务正常运行，无异常报错。</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}