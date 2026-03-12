import React from 'react';
import { Icon } from '@iconify/react';

type LayoutProps = {
    activeTab: string;
    setActiveTab: (tab: string) => void;
    children: React.ReactNode;
};

export default function Layout({ activeTab, setActiveTab, children }: LayoutProps) {
    return (
        <div className="flex h-screen bg-base-200 text-base-content overflow-hidden">
            {/* Narrow Sidebar */}
            <div className="w-16 bg-base-300 flex flex-col items-center py-4 border-r border-base-100 shadow-sm z-10 flex-shrink-0">

                {/* Top Logo */}
                <div className="mb-6 tooltip tooltip-right z-50" data-tip="Context Sherpa">
                    <div className="avatar placeholder cursor-pointer" onClick={() => setActiveTab('home')}>
                        <div className="bg-primary text-primary-content rounded-xl w-10 h-10 flex items-center justify-center shadow-lg transition-transform hover:scale-105">
                            <Icon icon="lucide:mountain-snow" className="text-2xl" />
                        </div>
                    </div>
                </div>

                {/* Top Menu Items */}
                <div className="flex-1 flex flex-col gap-3 w-full px-2 mt-4">
                    <button
                        className={`btn btn-square btn-ghost w-full transition-colors ${activeTab === 'home' ? 'bg-base-100 shadow-sm text-primary' : 'text-base-content/70 hover:bg-base-100 hover:text-base-content'} tooltip tooltip-right`}
                        data-tip="Dashboard"
                        onClick={() => setActiveTab('home')}
                    >
                        <Icon icon="lucide:layout-dashboard" className="text-xl" />
                    </button>

                    <button
                        className={`btn btn-square btn-ghost w-full transition-colors ${activeTab === 'atlas' ? 'bg-base-100 shadow-sm text-primary' : 'text-base-content/70 hover:bg-base-100 hover:text-base-content'} tooltip tooltip-right`}
                        data-tip="Code Atlas"
                        onClick={() => setActiveTab('atlas')}
                    >
                        <Icon icon="lucide:network" className="text-xl" />
                    </button>
                </div>

                {/* Bottom Menu Items */}
                <div className="w-full px-2 mt-auto">
                    <button
                        className={`btn btn-square btn-ghost w-full transition-colors ${activeTab === 'settings' ? 'bg-base-100 shadow-sm text-primary' : 'text-base-content/70 hover:bg-base-100 hover:text-base-content'} tooltip tooltip-right`}
                        data-tip="Settings"
                        onClick={() => setActiveTab('settings')}
                    >
                        <Icon icon="lucide:settings" className="text-xl" />
                    </button>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 overflow-hidden relative bg-base-200">
                <div className={`${activeTab === 'atlas' ? 'p-0 w-full overflow-hidden' : 'p-8 ml-4 overflow-y-auto'} h-full flex flex-col`}>
                    {children}
                </div>
            </div>
        </div>
    );
}
