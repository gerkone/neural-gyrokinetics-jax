// tex strings (json list on stdin) -> svg strings (json list on stdout), paths inlined
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', {InputJax: new TeX({packages: AllPackages}), OutputJax: new SVG({fontCache: 'none'})});
let input = '';
process.stdin.on('data', d => input += d);
process.stdin.on('end', () => {
  const out = JSON.parse(input).map(t => adaptor.innerHTML(doc.convert(t, {display: false})));
  process.stdout.write(JSON.stringify(out));
});
