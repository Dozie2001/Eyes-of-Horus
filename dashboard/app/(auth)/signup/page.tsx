"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Eye } from "lucide-react";

export default function SignupPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [orgName, setOrgName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const router = useRouter();

  async function handleSignup(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    const supabase = createClient();

    // 1. Create auth user
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

    // 2. Create organization + membership via RPC
    // (Uses service role on the backend — for now, create org client-side
    //  since RLS INSERT policies aren't set up yet. This will be moved
    //  to a server action or cloud API endpoint.)
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
      // If org creation fails, the user still exists in auth
      // They can create an org later or be invited to one
      console.error("Org creation failed:", orgError);
      setSuccess(true);
      setLoading(false);
      return;
    }

    // 3. Add user as owner of the org
    await supabase.from("org_member").insert({
      org_id: org.id,
      user_id: authData.user.id,
      role: "owner",
    });

    setSuccess(true);
    setLoading(false);
  }

  if (success) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="w-full max-w-sm space-y-6 px-6 text-center">
          <div className="flex items-center justify-center gap-2 text-horus">
            <Eye className="h-8 w-8" />
          </div>
          <h1 className="font-mono text-xl tracking-[0.15em] uppercase text-foreground">
            Check your email
          </h1>
          <p className="text-sm text-muted-foreground">
            We sent a confirmation link to <strong>{email}</strong>.
            Click it to activate your account.
          </p>
          <Link href="/login">
            <Button variant="outline" className="mt-4">
              Back to login
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="w-full max-w-sm space-y-8 px-6">
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2 text-horus">
            <Eye className="h-8 w-8" />
          </div>
          <h1 className="font-mono text-xl tracking-[0.15em] uppercase text-foreground">
            Eyes of Horus
          </h1>
          <p className="text-sm text-muted-foreground">
            Create your account
          </p>
        </div>

        <form onSubmit={handleSignup} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="orgName">Organization name</Label>
            <Input
              id="orgName"
              type="text"
              placeholder="Acme Warehouses"
              value={orgName}
              onChange={(e) => setOrgName(e.target.value)}
              required
              autoFocus
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="you@company.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              placeholder="Min 6 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
            />
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

        <p className="text-center text-sm text-muted-foreground">
          Already have an account?{" "}
          <Link href="/login" className="text-horus hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
