"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Eye, EyeIcon, EyeOffIcon } from "lucide-react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const router = useRouter();

  async function handleLogin(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setLoading(true);

    const supabase = createClient();
    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      setError(error.message);
      setLoading(false);
      return;
    }

    router.push("/");
    router.refresh();
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="flex w-full max-w-[440px] shadow-none flex-col gap-6 p-5 md:p-8">
        <CardHeader className="flex flex-col items-center gap-2">
          <div className="relative flex size-[68px] shrink-0 items-center justify-center rounded-full backdrop-blur-xl md:size-24 before:absolute before:inset-0 before:rounded-full before:bg-gradient-to-b before:from-horus/30 before:to-transparent before:opacity-30">
            <div className="relative z-10 flex size-12 items-center justify-center rounded-full bg-background dark:bg-muted/80 shadow-xs ring-1 ring-inset ring-border md:size-16">
              <Eye className="size-6 text-horus md:size-8" />
            </div>
          </div>

          <div className="flex flex-col space-y-1.5 text-center">
            <p className="font-mono text-[10px] tracking-[0.3em] uppercase text-horus">
              eyes of horus
            </p>
            <CardTitle className="md:text-xl font-medium">
              Sign in to your account
            </CardTitle>
            <CardDescription className="tracking-[-0.006em]">
              Enter your credentials to access your dashboard.
            </CardDescription>
          </div>
        </CardHeader>

        <Separator />

        <CardContent className="p-0">
          <form onSubmit={handleLogin} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2.5">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="you@company.com"
                className="rounded-lg"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoFocus
              />
            </div>

            <div className="flex flex-col gap-2.5">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  className="pe-9 rounded-lg"
                  placeholder="Your password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <button
                  className="text-muted-foreground/80 hover:text-foreground focus-visible:border-ring focus-visible:ring-ring/50 absolute inset-y-0 end-0 flex h-full w-9 items-center justify-center rounded-e-md transition-[color,box-shadow] outline-none focus:z-10 focus-visible:ring-[3px]"
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? (
                    <EyeOffIcon size={16} aria-hidden="true" />
                  ) : (
                    <EyeIcon size={16} aria-hidden="true" />
                  )}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between gap-4">
              <div className="flex items-start gap-2">
                <Checkbox id="keep-logged-in" />
                <Label htmlFor="keep-logged-in" className="block cursor-pointer">
                  Keep me logged in
                </Label>
              </div>
            </div>

            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <Button
              type="submit"
              className="w-full bg-horus text-background hover:bg-horus/90"
              disabled={loading}
            >
              {loading ? "Signing in..." : "Sign in"}
            </Button>
          </form>
        </CardContent>

        <Separator />

        <p className="text-center text-sm text-muted-foreground">
          No account?{" "}
          <Link href="/signup" className="text-horus hover:underline">
            Create one
          </Link>
        </p>
      </Card>
    </div>
  );
}
