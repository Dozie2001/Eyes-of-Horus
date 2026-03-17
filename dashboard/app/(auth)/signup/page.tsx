"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Eye, EyeIcon, EyeOffIcon, Building2, CheckCircle2 } from "lucide-react";

export default function SignupPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [orgName, setOrgName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const router = useRouter();

  async function handleSignup(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setLoading(true);

    const supabase = createClient();

    const { data: authData, error: authError } = await supabase.auth.signUp({
      email,
      password,
    });

    if (authError) {
      setError(authError.message);
      setLoading(false);
      return;
    }

    if (!authData.user) {
      setError("Signup failed — no user returned");
      setLoading(false);
      return;
    }

    const slug = orgName
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");

    const { data: org, error: orgError } = await supabase
      .from("organization")
      .insert({ name: orgName, slug })
      .select("id")
      .single();

    if (orgError) {
      console.error("Org creation failed:", orgError);
    } else {
      await supabase.from("org_member").insert({
        org_id: org.id,
        user_id: authData.user.id,
        role: "owner",
      });
    }

    setSuccess(true);
    setLoading(false);
  }

  if (success) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-4">
        <Card className="flex w-full max-w-[440px] shadow-none flex-col gap-6 p-5 md:p-8">
          <CardHeader className="flex flex-col items-center gap-4">
            <div className="relative flex size-[68px] shrink-0 items-center justify-center rounded-full backdrop-blur-xl md:size-24 before:absolute before:inset-0 before:rounded-full before:bg-gradient-to-b before:from-green-500/30 before:to-transparent before:opacity-30">
              <div className="relative z-10 flex size-12 items-center justify-center rounded-full bg-background dark:bg-muted/80 shadow-xs ring-1 ring-inset ring-border md:size-16">
                <CheckCircle2 className="size-6 text-green-500 md:size-8" />
              </div>
            </div>

            <div className="flex flex-col space-y-1.5 text-center">
              <CardTitle className="md:text-xl font-medium">
                Check your email
              </CardTitle>
              <CardDescription className="tracking-[-0.006em]">
                We sent a confirmation link to <strong className="text-foreground">{email}</strong>.
                Click it to activate your account.
              </CardDescription>
            </div>
          </CardHeader>

          <Separator />

          <div className="text-center">
            <Link href="/login">
              <Button variant="outline" className="w-full">
                Back to login
              </Button>
            </Link>
          </div>
        </Card>
      </div>
    );
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
              Create your account
            </CardTitle>
            <CardDescription className="tracking-[-0.006em]">
              Set up your organization to start monitoring.
            </CardDescription>
          </div>
        </CardHeader>

        <Separator />

        <CardContent className="p-0">
          <form onSubmit={handleSignup} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2.5">
              <Label htmlFor="orgName">Organization name</Label>
              <div className="relative">
                <Input
                  id="orgName"
                  type="text"
                  placeholder="Acme Warehouses"
                  className="ps-9 rounded-lg"
                  value={orgName}
                  onChange={(e) => setOrgName(e.target.value)}
                  required
                  autoFocus
                />
                <div className="absolute inset-y-0 start-0 flex items-center ps-3 text-muted-foreground/60">
                  <Building2 size={16} />
                </div>
              </div>
            </div>

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
              />
            </div>

            <div className="flex flex-col gap-2.5">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  className="pe-9 rounded-lg"
                  placeholder="Min 6 characters"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
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

            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <Button
              type="submit"
              className="w-full bg-horus text-background hover:bg-horus/90"
              disabled={loading}
            >
              {loading ? "Creating account..." : "Create account"}
            </Button>
          </form>
        </CardContent>

        <Separator />

        <p className="text-center text-sm text-muted-foreground">
          Already have an account?{" "}
          <Link href="/login" className="text-horus hover:underline">
            Sign in
          </Link>
        </p>
      </Card>
    </div>
  );
}
