import { describe, expect, it } from "vitest";

import { buildSuperSplatEditorUrl } from "../src/viewer/ExternalViewerLink";

describe("buildSuperSplatEditorUrl", () => {
  it("builds the official load query only when a browser-reachable base URL exists", () => {
    expect(
      buildSuperSplatEditorUrl({
        artifactUrl: "/api/v1/jobs/job-1/artifacts/published-ply",
        publicBaseUrl: "http://127.0.0.1:8000",
      }),
    ).toBe(
      "https://superspl.at/editor?load=http%3A%2F%2F127.0.0.1%3A8000%2Fapi%2Fv1%2Fjobs%2Fjob-1%2Fartifacts%2Fpublished-ply",
    );
  });

  it("returns null when the local artifact has no externally reachable URL", () => {
    expect(
      buildSuperSplatEditorUrl({
        artifactUrl: "/api/v1/jobs/job-1/artifacts/published-ply",
      }),
    ).toBeNull();
  });
});
