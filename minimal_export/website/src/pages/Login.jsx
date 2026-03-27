import React from 'react';
import { Link } from 'react-router-dom';

export default function Login() {
  return (
    <div className="flex min-h-[70vh] flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <h2 className="mt-6 text-center text-3xl font-bold tracking-tight text-slate-200">
          登录您的账户
        </h2>
        <p className="mt-2 text-center text-sm text-slate-400">
          或者{' '}
          <a href="#" className="font-medium text-primary hover:text-blue-500">
            注册一个新账户
          </a>
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-slate-900/50 py-8 px-4 shadow sm:rounded-2xl border border-slate-800 sm:px-10">
          <form className="space-y-6" action="#" method="POST">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-slate-300">
                邮箱地址
              </label>
              <div className="mt-1">
                <input id="email" name="email" type="email" autoComplete="email" required aria-label="邮箱地址" className="block w-full appearance-none rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-200 placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary sm:text-sm" />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-slate-300">
                密码
              </label>
              <div className="mt-1">
                <input id="password" name="password" type="password" autoComplete="current-password" required aria-label="密码" className="block w-full appearance-none rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-200 placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary sm:text-sm" />
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input id="remember-me" name="remember-me" type="checkbox" className="h-4 w-4 rounded border-slate-700 text-primary focus:ring-primary bg-slate-800" />
                <label htmlFor="remember-me" className="ml-2 block text-sm text-slate-400">
                  记住我
                </label>
              </div>

              <div className="text-sm">
                <a href="#" className="font-medium text-primary hover:text-blue-500">
                  忘记密码?
                </a>
              </div>
            </div>

            <div>
              <Link to="/admin" className="flex w-full justify-center rounded-md border border-transparent bg-primary py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-slate-900 transition-colors">
                登录 (Mock)
              </Link>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}