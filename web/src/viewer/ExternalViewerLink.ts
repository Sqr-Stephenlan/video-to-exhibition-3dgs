const DEFAULT_VIEWER_BASE = "https://superspl.at/editor";

export interface ExternalViewerLinkOptions {
  artifactUrl: string;
  publicBaseUrl?: string;
  viewerBaseUrl?: string;
}

export function buildSuperSplatEditorUrl({
  artifactUrl,
  publicBaseUrl,
  viewerBaseUrl = DEFAULT_VIEWER_BASE,
}: ExternalViewerLinkOptions): string | null {
  if (!publicBaseUrl) {
    return null;
  }
  let modelUrl: URL;
  let viewerUrl: URL;
  try {
    modelUrl = new URL(artifactUrl, publicBaseUrl);
    viewerUrl = new URL(viewerBaseUrl);
  } catch {
    return null;
  }
  if (!["http:", "https:"].includes(modelUrl.protocol)) {
    return null;
  }
  if (!["http:", "https:"].includes(viewerUrl.protocol)) {
    return null;
  }
  viewerUrl.searchParams.set("load", modelUrl.toString());
  return viewerUrl.toString();
}
