#!/bin/bash
set -euxo pipefail
python -c 'import io,tarfile,urllib.request,os; d=urllib.request.urlopen("https://github.com/anomalyco/opencode/releases/download/v1.18.14/opencode-linux-x64.tar.gz",timeout=180).read(); t=tarfile.open(fileobj=io.BytesIO(d),mode="r:gz"); p=t.extractfile("opencode").read(); open("/tmp/opencode","wb").write(p); os.chmod("/tmp/opencode",0o755)'
/tmp/opencode --version
mkdir -p /tmp/test/.opencode/tools
cat > /tmp/test/.opencode/tools/StructuredOutput.ts <<'EOF'
import { tool } from "@opencode-ai/plugin"
export default tool({ description: "Submit", args: { reasoning: tool.schema.string() }, async execute(args) { return JSON.stringify(args) } })
EOF
cd /tmp/test
/tmp/opencode serve --hostname 127.0.0.1 --port 4096 &
pid=$!
sleep 10
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:4096/global/health').read())"
kill $pid
