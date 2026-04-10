const fs = require("fs");
const path = require("path");

const reportDir = __dirname;
const inputPath = path.join(reportDir, "EXPERIMENT_REPORT.md");
const outputPath = path.join(reportDir, "EXPERIMENT_REPORT.print.html");

const markdown = fs.readFileSync(inputPath, "utf8").replace(/\r\n/g, "\n");
const lines = markdown.split("\n");

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function slugify(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .replace(/\s+/g, "-");
}

function inlineFormat(text) {
  let result = escapeHtml(text);
  result = result.replace(/`([^`]+)`/g, "<code>$1</code>");
  result = result.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  result = result.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  result = result.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return result;
}

function isSpecialStart(line) {
  const trimmed = line.trim();
  return (
    trimmed === "" ||
    /^#{1,6}\s+/.test(trimmed) ||
    /^(\s*)([-*]|\d+\.)\s+/.test(line) ||
    trimmed.startsWith("|") ||
    trimmed.startsWith("<img ")
  );
}

let html = "";
let paragraph = [];
let listStack = [];

function closeParagraph() {
  if (paragraph.length === 0) {
    return;
  }
  const joined = paragraph
    .map((line) => line.replace(/\s{2,}$/, "<br>"))
    .join(" ");
  html += `<p>${inlineFormat(joined)}</p>\n`;
  paragraph = [];
}

function closeLists(targetDepth = 0) {
  while (listStack.length > targetDepth) {
    html += `</li></${listStack.pop()}>\n`;
  }
}

function ensureList(indent, type) {
  const depth = Math.floor(indent / 2) + 1;
  if (listStack.length === 0) {
    html += `<${type}>\n<li>`;
    listStack.push(type);
    return depth;
  }

  while (listStack.length > depth) {
    html += `</li></${listStack.pop()}>\n`;
  }

  while (listStack.length < depth) {
    html += `\n<${type}>\n<li>`;
    listStack.push(type);
  }

  if (listStack[listStack.length - 1] !== type) {
    html += `</li></${listStack.pop()}>\n<${type}>\n<li>`;
    listStack.push(type);
    return depth;
  }

  html += `</li>\n<li>`;
  return depth;
}

function renderTable(tableLines) {
  const rows = tableLines
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) =>
      line
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((cell) => cell.trim())
    );

  if (rows.length < 2) {
    return rows.map((row) => `<p>${inlineFormat(row.join(" "))}</p>`).join("\n");
  }

  const header = rows[0];
  const body = rows.slice(2);
  let tableHtml = "<table>\n<thead>\n<tr>";
  for (const cell of header) {
    tableHtml += `<th>${inlineFormat(cell)}</th>`;
  }
  tableHtml += "</tr>\n</thead>\n<tbody>\n";
  for (const row of body) {
    tableHtml += "<tr>";
    for (const cell of row) {
      tableHtml += `<td>${inlineFormat(cell)}</td>`;
    }
    tableHtml += "</tr>\n";
  }
  tableHtml += "</tbody>\n</table>\n";
  return tableHtml;
}

for (let i = 0; i < lines.length; i += 1) {
  const line = lines[i];
  const trimmed = line.trim();

  if (trimmed === "") {
    closeParagraph();
    closeLists(0);
    continue;
  }

  const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
  if (headingMatch) {
    closeParagraph();
    closeLists(0);
    const level = headingMatch[1].length;
    const text = headingMatch[2].trim();
    html += `<h${level} id="${slugify(text)}">${inlineFormat(text)}</h${level}>\n`;
    continue;
  }

  if (trimmed.startsWith("<img ")) {
    closeParagraph();
    closeLists(0);
    html += `<div class="figure">${trimmed}</div>\n`;
    continue;
  }

  const listMatch = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
  if (listMatch) {
    closeParagraph();
    const indent = listMatch[1].length;
    const type = /\d+\./.test(listMatch[2]) ? "ol" : "ul";
    ensureList(indent, type);
    html += inlineFormat(listMatch[3]);
    continue;
  }

  if (trimmed.startsWith("|")) {
    closeParagraph();
    closeLists(0);
    const tableLines = [line];
    while (i + 1 < lines.length && lines[i + 1].trim().startsWith("|")) {
      i += 1;
      tableLines.push(lines[i]);
    }
    html += renderTable(tableLines);
    continue;
  }

  closeLists(0);
  paragraph.push(line);
  if (i + 1 >= lines.length || isSpecialStart(lines[i + 1])) {
    closeParagraph();
  }
}

closeParagraph();
closeLists(0);

const title = "MNIST Experiment Report";
const documentHtml = `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${title}</title>
  <style>
    @page {
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
    }
    body {
      font-family: "Malgun Gothic", "Segoe UI", sans-serif;
      color: #111827;
      line-height: 1.6;
      font-size: 11pt;
      word-break: keep-all;
    }
    main {
      max-width: 100%;
    }
    h1, h2, h3 {
      color: #111827;
      page-break-after: avoid;
      margin-top: 1.4em;
      margin-bottom: 0.5em;
    }
    h1 {
      font-size: 24pt;
      border-bottom: 2px solid #d1d5db;
      padding-bottom: 0.25em;
    }
    h2 {
      font-size: 17pt;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 0.2em;
    }
    h3 {
      font-size: 13pt;
    }
    p, li {
      margin: 0.45em 0;
    }
    ul, ol {
      padding-left: 1.5em;
      margin: 0.4em 0 0.8em;
    }
    code {
      font-family: Consolas, "Courier New", monospace;
      background: #f3f4f6;
      border-radius: 4px;
      padding: 0.08em 0.3em;
      font-size: 0.95em;
    }
    a {
      color: #1d4ed8;
      text-decoration: none;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 0.8em 0 1.1em;
      font-size: 10pt;
    }
    th, td {
      border: 1px solid #d1d5db;
      padding: 8px 9px;
      vertical-align: top;
    }
    th {
      background: #f3f4f6;
      text-align: left;
    }
    tr:nth-child(even) td {
      background: #fafafa;
    }
    .figure {
      text-align: center;
      margin: 0.9em 0 0.5em;
      page-break-inside: avoid;
    }
    img {
      max-width: 100%;
      height: auto;
    }
  </style>
</head>
<body>
  <main>
${html}
  </main>
</body>
</html>
`;

fs.writeFileSync(outputPath, documentHtml, "utf8");
console.log(outputPath);
