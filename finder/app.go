package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"runtime"
	"github.com/tyler-smith/go-bip39"
)

const (
	stateFile    = "generator_state.txt"
	mnemonicsFile = "mnemonics.txt"
)

type Generator struct {
	mu         sync.Mutex
	index      uint64
	file       *os.File
	shutdown   chan struct{}
	wg         sync.WaitGroup
}

func NewGenerator() *Generator {
	f, err := os.OpenFile(mnemonicsFile, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		log.Fatal(err)
	}

	var initialIndex uint64
	if state, err := os.ReadFile(stateFile); err == nil {
		fmt.Sscanf(string(state), "%d", &initialIndex)
	}

	return &Generator{
		index:    initialIndex,
		file:     f,
		shutdown: make(chan struct{}),
	}
}

func (g *Generator) generateWorker() {
	defer g.wg.Done()
	
	batch := make([]string, 0, 1000)
	ticker := time.NewTicker(1 * time.Second)
	
	for {
		select {
		case <-g.shutdown:
			g.flushBatch(batch)
			return
		case <-ticker.C:
			if len(batch) > 0 {
				g.flushBatch(batch)
				batch = make([]string, 0, 1000)
			}
		default:
			entropy, _ := bip39.NewEntropy(256)
			mnemonic, _ := bip39.NewMnemonic(entropy)
			batch = append(batch, mnemonic)
			atomic.AddUint64(&g.index, 1)
		}
	}
}

func (g *Generator) flushBatch(batch []string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	for _, m := range batch {
		if _, err := g.file.WriteString(m + "\n"); err != nil {
			log.Printf("Write error: %v", err)
		}
	}
	
	if err := g.file.Sync(); err != nil {
		log.Printf("Sync error: %v", err)
	}
}

func (g *Generator) Run(workers int) {
	for i := 0; i < workers; i++ {
		g.wg.Add(1)
		go g.generateWorker()
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	
	<-sig
	log.Println("Shutting down...")
	close(g.shutdown)
	g.wg.Wait()
	
	g.saveState()
	g.file.Close()
}

func (g *Generator) saveState() {
	if f, err := os.Create(stateFile); err == nil {
		fmt.Fprintf(f, "%d", atomic.LoadUint64(&g.index))
		f.Close()
	}
}

func main() {
	generator := NewGenerator()
	log.Println("Starting mnemonic generator...")
	generator.Run(runtime.NumCPU())
}