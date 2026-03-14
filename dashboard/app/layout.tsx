import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { SidebarTrigger } from "@/components/ui/sidebar";
import { Providers } from "@/components/providers";
import { AppSidebar } from "@/components/app-sidebar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Eyes of Horus",
  description: "AI CCTV Monitoring Dashboard",
};

/**
 * Root layout — wraps every page in the app.
 *
 * Structure:
 *   Providers (QueryClient + Sidebar + Tooltip + Toaster)
 *     ├── AppSidebar (left nav, collapses on mobile)
 *     └── main content area
 *           ├── SidebarTrigger (hamburger button, visible on mobile)
 *           └── {children} (the current page)
 *
 * Providers is a client component that bundles all context providers.
 * This layout stays as a server component (better for performance).
 */
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <Providers>
          <AppSidebar />
          <main className="flex-1 overflow-auto">
            <div className="flex items-center gap-2 border-b p-2 md:hidden">
              <SidebarTrigger />
              <span className="text-sm font-medium">Eyes of Horus</span>
            </div>
            <div className="p-4 md:p-6">
              {children}
            </div>
          </main>
        </Providers>
      </body>
    </html>
  );
}
