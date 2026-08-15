import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const ENGINE_ORIGIN = process.env.AMMA_ENGINE_ORIGIN ?? "http://127.0.0.1:7860";
const WEB_ORIGIN = process.env.AMMA_WEB_ORIGIN ?? "http://127.0.0.1:3000";

function requestIsLocal(request: NextRequest) {
  const fetchSite = request.headers.get("sec-fetch-site");
  if (fetchSite === "cross-site") return false;
  const origin = request.headers.get("origin");
  return !origin || origin === WEB_ORIGIN;
}

async function forward(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  if (!requestIsLocal(request)) {
    return Response.json({ detail: "Cross-site requests are not allowed." }, { status: 403 });
  }

  const token = process.env.AMMA_LOCAL_TOKEN;
  if (!token) {
    return Response.json({ detail: "Start the app with Open amma.live Music.command." }, { status: 503 });
  }

  const { path } = await context.params;
  if (!path.length || path.some((segment) => !/^[A-Za-z0-9._-]+$/.test(segment))) {
    return Response.json({ detail: "Invalid local path." }, { status: 400 });
  }

  const enginePath = path[0] === "media" ? `/${path.join("/")}` : `/api/${path.join("/")}`;
  const headers = new Headers({ Authorization: `Bearer ${token}` });
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("Content-Type", contentType);
  const range = request.headers.get("range");
  if (range) headers.set("Range", range);

  try {
    const upstream = await fetch(`${ENGINE_ORIGIN}${enginePath}`, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.arrayBuffer(),
      cache: "no-store",
      redirect: "error",
    });
    const responseHeaders = new Headers({
      "Cache-Control": "no-store",
      "Content-Type": upstream.headers.get("content-type") ?? "application/octet-stream",
      "X-Content-Type-Options": "nosniff",
    });
    const disposition = upstream.headers.get("content-disposition");
    if (disposition) responseHeaders.set("Content-Disposition", disposition);
    for (const name of ["accept-ranges", "content-length", "content-range"]) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch {
    return Response.json(
      { detail: "The private music engine is not running. Keep the local app window open and try again." },
      { status: 503 },
    );
  }
}

export const GET = forward;
export const HEAD = forward;
export const POST = forward;
