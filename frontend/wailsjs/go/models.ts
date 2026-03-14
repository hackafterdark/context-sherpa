export namespace inference {
	
	export class ModelInfo {
	    id: string;
	    name: string;
	    size: number;
	    path: string;
	    downloaded: boolean;
	    downloadUrl: string;
	    description: string;
	    lastUsed: string;
	
	    static createFrom(source: any = {}) {
	        return new ModelInfo(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.name = source["name"];
	        this.size = source["size"];
	        this.path = source["path"];
	        this.downloaded = source["downloaded"];
	        this.downloadUrl = source["downloadUrl"];
	        this.description = source["description"];
	        this.lastUsed = source["lastUsed"];
	    }
	}

}

export namespace main {
	
	export class CyElement {
	    group: string;
	    data: any;
	
	    static createFrom(source: any = {}) {
	        return new CyElement(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.group = source["group"];
	        this.data = source["data"];
	    }
	}
	export class GraphCategory {
	    name: string;
	
	    static createFrom(source: any = {}) {
	        return new GraphCategory(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	    }
	}
	export class GraphData {
	    elements: CyElement[];
	    categories: GraphCategory[];
	    language: string;
	
	    static createFrom(source: any = {}) {
	        return new GraphData(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.elements = this.convertValues(source["elements"], CyElement);
	        this.categories = this.convertValues(source["categories"], GraphCategory);
	        this.language = source["language"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class MarkdownEntry {
	    path: string;
	    frontMatter: Record<string, string>;
	
	    static createFrom(source: any = {}) {
	        return new MarkdownEntry(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.path = source["path"];
	        this.frontMatter = source["frontMatter"];
	    }
	}
	export class UserPreferences {
	    theme: string;
	    windowWidth: number;
	    windowHeight: number;
	    windowX: number;
	    windowY: number;
	    isMaximized: boolean;
	    inferenceProvider: string;
	    inferenceURL: string;
	    inferenceModel: string;
	
	    static createFrom(source: any = {}) {
	        return new UserPreferences(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.theme = source["theme"];
	        this.windowWidth = source["windowWidth"];
	        this.windowHeight = source["windowHeight"];
	        this.windowX = source["windowX"];
	        this.windowY = source["windowY"];
	        this.isMaximized = source["isMaximized"];
	        this.inferenceProvider = source["inferenceProvider"];
	        this.inferenceURL = source["inferenceURL"];
	        this.inferenceModel = source["inferenceModel"];
	    }
	}
	export class Workspace {
	    pid: number;
	    root: string;
	    client: string;
	    state: string;
	    lastSeen: string;
	    isManaged: boolean;
	
	    static createFrom(source: any = {}) {
	        return new Workspace(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.pid = source["pid"];
	        this.root = source["root"];
	        this.client = source["client"];
	        this.state = source["state"];
	        this.lastSeen = source["lastSeen"];
	        this.isManaged = source["isManaged"];
	    }
	}

}

