const beautify = require('js-beautify').js;

function deobfuscateAndFormat(code) {
  let decodedCode = '';
  const hexRegex = /_0x[a-fA-F0-9]+/g;
  const hexMap = {};
  let match;
  
  while ((match = hexRegex.exec(code)) !== null) {
    const hex = match[0];
    if (!hexMap[hex]) {
      hexMap[hex] = `hex_${Object.keys(hexMap).length}`;
    }
  }
  
  for (const key in hexMap) {
    code = code.split(key).join(hexMap[key]);
  }
  
  code = code
    .split('\n')
    .map(line => line.trim())
    .join('\n');
  
  const lineRegex = /;/g;
  let lastIndex = 0;
  
  while ((match = lineRegex.exec(code)) !== null) {
    decodedCode += code.substring(lastIndex, match.index + 1) + '\n';
    lastIndex = match.index + 1;
  }
  
  decodedCode += code.substring(lastIndex);
  
  return beautify(decodedCode, { indent_size: 2 });
}
