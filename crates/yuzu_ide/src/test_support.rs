use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use text_size::TextSize;

use crate::{Analysis, AnalysisHost, Change, Checked, FileId, FilePosition};

pub(crate) const FILE: FileId = FileId(0);

pub(crate) fn analysis(text: &str) -> Analysis {
    let mut change = Change::default();
    change.set_file(FILE, Some(text.into()));
    let mut host = AnalysisHost::default();
    host.apply_change(change);
    host.analysis()
}

pub(crate) struct Tree(pub(crate) PathBuf);

impl Tree {
    pub(crate) fn new(files: &[(&str, &str)]) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "yuzu-ide-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        for (path, text) in files {
            let file = root.join(path);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, text).unwrap();
        }
        Tree(root)
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn cursor(fixture: &str) -> (String, TextSize) {
    let offset = fixture
        .find("$0")
        .expect("the fixture marks a position with $0");
    let offset = TextSize::try_from(offset).expect("a fixture is shorter than 4 GiB");
    (fixture.replacen("$0", "", 1), offset)
}

pub(crate) fn at(offset: TextSize) -> FilePosition {
    FilePosition {
        file_id: FILE,
        offset,
    }
}

pub(crate) fn checked(files: &[(&str, &str)], text: &str) -> (Tree, Checked) {
    let tree = Tree::new(files);
    let main = tree.0.join("main.yz");
    let mut change = Change::default();
    change.set_file(FILE, Some(text.into()));
    change.set_path(FILE, Some(main));
    let mut host = AnalysisHost::default();
    host.apply_change(change);
    let checked = host.analysis().check(FILE).expect("main.yz has a path");
    (tree, checked)
}

pub(crate) fn render(checked: &Checked, path: &Path, range: text_size::TextRange) -> String {
    let file = path.file_name().unwrap().to_string_lossy();
    let text = checked.path_text(path).unwrap();
    format!("{file}:{}", &text[range])
}
