// Uses an installed Chromium browser and Node's built-in CDP WebSocket client.
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "demo", "validation");
await fs.mkdir(output, { recursive: true });
const profile = path.join(root, ".demo-browser", `check-${Date.now()}`);
await fs.mkdir(profile, { recursive: true });
const serverLog = await fs.open(path.join(output, "browser-server.log"), "w");
const bundledPython = path.join(root, ".python-runtime", "python.exe");
const python = await fs.access(bundledPython).then(() => bundledPython,
  () => path.join(root, ".venv", "Scripts", "python.exe"));
const server = spawn(python,
  ["-X", "utf8", path.join(root, "demo", "server.py"), "--port", "8767"],
  { cwd: root, stdio: ["ignore", serverLog.fd, serverLog.fd], windowsHide: true });
let chrome, socket;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, timeout = 30000) {
  const end = Date.now() + timeout;
  let error;
  while (Date.now() < end) {
    try { const result = await fn(); if (result) return result; } catch (e) { error = e; }
    await pause(150);
  }
  throw new Error(`Timeout: ${error?.message || "condition not met"}`);
}
try {
  await until(async () => (await fetch("http://127.0.0.1:8767/api/status")).ok);
  chrome = spawn("C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
     "--disable-background-networking", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"],
    { stdio: "ignore", windowsHide: true });
  chrome.on("error", error => console.error(error.message));
  const port = await until(async () => Number((await fs.readFile(path.join(profile, "DevToolsActivePort"), "utf8")).split("\n")[0]));
  const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  socket = new WebSocket(targets.find(target => target.type === "page").webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.addEventListener("open", resolve); socket.addEventListener("error", reject); });
  const pending = new Map(); let sequence = 0; const exceptions = [];
  socket.addEventListener("message", message => {
    const event = JSON.parse(message.data);
    if (event.id && pending.has(event.id)) {
      const { resolve, reject } = pending.get(event.id); pending.delete(event.id);
      if (event.error) reject(new Error(JSON.stringify(event.error))); else resolve(event.result);
    } else if (event.method === "Runtime.exceptionThrown") exceptions.push(event.params);
  });
  function call(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++sequence; pending.set(id, { resolve, reject });
      socket.send(JSON.stringify({ id, method, params }));
    });
  }
  async function evaluate(expression) {
    const response = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (response.exceptionDetails) throw new Error(response.exceptionDetails.text);
    return response.result?.value;
  }
  async function screenshot(name) {
    const { data } = await call("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
    await fs.writeFile(path.join(output, name), Buffer.from(data, "base64"));
  }
  await call("Page.enable"); await call("Runtime.enable");
  await call("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
  await call("Page.navigate", { url: "http://127.0.0.1:8767" });
  await until(() => evaluate("document.readyState === 'complete' && document.getElementById('modelStatus').textContent.includes('SẴN SÀNG')"));
  await screenshot("desktop-ready.png");
  const doc = await call("DOM.getDocument");
  const { nodeId } = await call("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#fileInput" });
  await call("DOM.setFileInputFiles", { nodeId, files: [path.join(root, "demo", "vendor", "semseg", "figure", "demo", "ADE_val_00001515.jpg")] });
  await until(() => evaluate("!document.getElementById('runButton').disabled"));
  await evaluate("document.getElementById('runButton').click()");
  await until(() => evaluate("!document.getElementById('activationImage').hidden"), 120000);
  await screenshot("desktop-processing.png");
  await until(() => evaluate("!document.getElementById('resultsPanel').hidden"), 300000);
  if (!await evaluate("document.getElementById('errorBox').hidden")) throw new Error(await evaluate("document.getElementById('errorBox').textContent"));
  await screenshot("desktop-result.png");
  await evaluate("document.getElementById('maskButton').click(); document.getElementById('compareSlider').value = 50; document.getElementById('compareSlider').dispatchEvent(new Event('input'))");
  const checks = await evaluate("({mask:document.getElementById('maskButton').classList.contains('active'),compare:document.getElementById('resultImage').style.clipPath,legend:document.getElementById('legend').children.length,download:document.getElementById('downloadLabels').href.startsWith('data:image/png;base64,'),caption:document.getElementById('resultCaption').textContent})");
  if (!checks.mask || !checks.compare.includes("50%") || !checks.legend || !checks.download) throw new Error(JSON.stringify(checks));
  await call("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await pause(300); await screenshot("mobile.png");
  const overflow = await evaluate("document.documentElement.scrollWidth > window.innerWidth");
  if (overflow) throw new Error("Mobile horizontal overflow");
  if (exceptions.length) throw new Error(JSON.stringify(exceptions));
  const report = { browser: "Chrome headless", desktop: "1440x1100", mobile: "390x844", exceptions,
    mobileOverflow: overflow, checks, passed: true };
  await fs.writeFile(path.join(output, "browser-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  socket?.close(); chrome?.kill(); server.kill(); await serverLog.close();
}
