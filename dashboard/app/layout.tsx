import type { Metadata } from "next";
import { IBM_Plex_Mono, DM_Sans } from "next/font/google";
import "./globals.css";

import { SidebarTrigger } from "@/components/ui/sidebar";
import { Providers } from "@/components/providers";
import { AppSidebar } from "@/components/app-sidebar";

const dmSans = DM_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["300", "400", "500"],
});

export const metadata: Metadata = {
  title: "Eyes of Horus",
  description: "AI CCTV Monitoring Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${dmSans.variable} ${ibmPlexMono.variable} antialiased`}
      >
        <Providers>
          <AppSidebar />
          <main className="flex-1 overflow-auto">
            <div className="flex items-center gap-3 border-b border-border/40 px-4 py-3 md:hidden">
              <SidebarTrigger />
              <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
                Eyes of Horus
              </span>
            </div>
            <div className="p-5 md:p-8 lg:p-10">
              {children}
            </div>
          </main>
        </Providers>
      </body>
    </html>
  );
}
