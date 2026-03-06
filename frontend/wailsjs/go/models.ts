export namespace main {
	
	export class Workspace {
	    pid: number;
	    root: string;
	    client: string;
	    state: string;
	
	    static createFrom(source: any = {}) {
	        return new Workspace(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.pid = source["pid"];
	        this.root = source["root"];
	        this.client = source["client"];
	        this.state = source["state"];
	    }
	}

}

